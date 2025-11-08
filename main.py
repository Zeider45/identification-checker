"""Endpoint único que combina la lógica de `qr_api.py` y `id-validator.py`.

Este archivo carga ambos scripts desde su ruta, reutiliza sus funciones
de procesamiento (detección de QR, procesamiento de PDF, OCR y validación)
y expone un endpoint POST `/validate_all` que acepta archivos multipart
con campos `cedula` y `rif` (uno o ambos) y devuelve una respuesta
unificada con los resultados de cada procesamiento.
"""
import os
import sys
import importlib.util
from typing import Dict, Any
from flask import Flask, request, jsonify
import shutil
import pytesseract as _pytesseract


def load_module_from_path(name: str, path: str):
    """Carga dinámicamente un módulo desde una ruta de archivo.

    name: nombre a asignar al módulo en tiempo de ejecución (solo interno).
    path: ruta absoluta al fichero .py
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar spec para {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE = os.path.dirname(__file__)
QR_PATH = os.path.join(BASE, 'qr_api.py')
IDV_PATH = os.path.join(BASE, 'id-validator.py')

# Cargar ambos módulos (si no existen, fallará aquí con excepción clara)
qr_mod = load_module_from_path('qr_api_mod', QR_PATH)
id_mod = load_module_from_path('id_validator_mod', IDV_PATH)

# Corregir la ruta a Tesseract si `id-validator.py` la fija a una ruta inexistente.
# Esto evita que OCR falle silenciosamente por una ruta absoluta de otra máquina.
try:
    tpath = getattr(id_mod, 'TESSERACT_CMD', None)
    if tpath and not os.path.exists(tpath):
        # buscar tesseract en PATH
        candidate = shutil.which('tesseract')
        if candidate:
            try:
                # reconfiguramos pytesseract dentro del módulo cargado
                id_mod.pytesseract.pytesseract.tesseract_cmd = candidate
                _pytesseract.pytesseract.tesseract_cmd = candidate
            except Exception:
                pass
        else:
            # si no se encuentra, aseguramos que la variable no apunte a ruta inválida
            try:
                id_mod.pytesseract.pytesseract.tesseract_cmd = ''
            except Exception:
                pass
except Exception:
    pass

app = Flask(__name__)


def process_bytes(file_bytes: bytes, filename: str, role: str) -> Dict[str, Any]:
    """Procesa un archivo (PDF o imagen) usando funciones de qr_mod e id_mod.

    role: etiqueta descriptiva ('cedula' o 'rif') usada en la respuesta.
    """
    out: Dict[str, Any] = {'role': role, 'filename': filename, 'type': None, 'qr': [], 'ocr_text': None, 'cedula_refined': None, 'validation': None, 'error': None}

    try:
        is_pdf = filename.lower().endswith('.pdf')
        if is_pdf:
            out['type'] = 'pdf'
            scan_res = qr_mod._scan_pdf(file_bytes)
            # separar QR y texto
            qr_items = [r for r in scan_res if isinstance(r, dict) and r.get('data')]
            pdf_texts = [r.get('pdf_text') for r in scan_res if isinstance(r, dict) and r.get('pdf_text')]
            if qr_items:
                out['qr'] = qr_items
                # seguir URLs dentro de los QR
                for r in out['qr']:
                    data_field = r.get('data')
                    if isinstance(data_field, str) and (data_field.startswith('http://') or data_field.startswith('https://')):
                        try:
                            r['remote'] = qr_mod._fetch_and_extract(data_field)
                        except Exception as e:
                            r['remote'] = {'error': str(e)}
            else:
                if pdf_texts:
                    out['ocr_text'] = '\n\n'.join([t for t in pdf_texts if t])
                    # intentar validación con id_mod si está disponible
                    try:
                        db_path = getattr(id_mod, 'DEFAULT_DB_PATH', os.path.join(BASE, 'database_example.json'))
                        threshold = getattr(id_mod, 'DEFAULT_THRESHOLD', 15.0)
                        out['validation'] = id_mod.validate_against_db(out['ocr_text'], db_path=db_path, threshold=threshold)
                    except Exception:
                        out['validation'] = None

        else:
            out['type'] = 'image'
            img = qr_mod._image_from_bytes(file_bytes)
            if img is None:
                out['error'] = 'invalid image'
                return out

            qr_items = qr_mod._detect_qr(img)
            if qr_items:
                out['qr'] = qr_items
                for r in out['qr']:
                    data_field = r.get('data')
                    if isinstance(data_field, str) and (data_field.startswith('http://') or data_field.startswith('https://')):
                        try:
                            r['remote'] = qr_mod._fetch_and_extract(data_field)
                        except Exception as e:
                            r['remote'] = {'error': str(e)}
            else:
                # sin QR: usar OCR y heurísticas del id-mod
                try:
                    out['ocr_text'] = id_mod.texto(img)
                except Exception:
                    out['ocr_text'] = None
                try:
                    ced_loc = id_mod.find_cedula_bbox_and_refine(img)
                    out['cedula_refined'] = ced_loc.get('cedula_refined') if isinstance(ced_loc, dict) else None
                    out['cedula_diag'] = ced_loc
                except Exception:
                    out['cedula_refined'] = None

                # intentar validar contra DB si hay texto
                if out.get('ocr_text'):
                    try:
                        db_path = getattr(id_mod, 'DEFAULT_DB_PATH', os.path.join(BASE, 'database_example.json'))
                        threshold = getattr(id_mod, 'DEFAULT_THRESHOLD', 15.0)
                        out['validation'] = id_mod.validate_against_db(out['ocr_text'], db_path=db_path, threshold=threshold, cedula_hint=out.get('cedula_refined'))
                    except Exception as e:
                        out['validation_error'] = str(e)

    except Exception as e:
        out['error'] = str(e)

    return out


@app.route('/validate_all', methods=['POST'])
def validate_all():
    """Endpoint que recibe archivos 'cedula' y/o 'rif' y devuelve resultados combinados."""
    if 'cedula' not in request.files and 'rif' not in request.files:
        return jsonify({'ok': False, 'error': "Se requiere al menos 'cedula' o 'rif' como archivos"}), 400

    resp: Dict[str, Any] = {'ok': True, 'results': {}}

    if 'cedula' in request.files:
        f = request.files['cedula']
        filename = (f.filename or 'cedula').lower()
        data = f.read()
        resp['results']['cedula'] = process_bytes(data, filename, role='cedula')

    if 'rif' in request.files:
        f = request.files['rif']
        filename = (f.filename or 'rif').lower()
        data = f.read()
        resp['results']['rif'] = process_bytes(data, filename, role='rif')

    return jsonify(resp), 200


if __name__ == '__main__':
    # Ejecutar servidor de desarrollo para pruebas locales
    port = 5000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except Exception:
            pass
    app.run(host='0.0.0.0', port=port, debug=True)

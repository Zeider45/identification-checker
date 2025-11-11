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
import re
import difflib


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

# Intentar configurar la ruta al ejecutable tesseract de forma robusta.
# Buscamos en PATH y en ubicaciones comunes de Windows y configuramos
# pytesseract en este proceso y en los módulos dinámicamente cargados.
try:
    candidate = shutil.which('tesseract')
    if not candidate:
        # rutas comunes en Windows
        possible = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for p in possible:
            if os.path.exists(p):
                candidate = p
                break

    if candidate:
        try:
            # configurar en el pytesseract importado aquí
            _pytesseract.pytesseract.tesseract_cmd = candidate
        except Exception:
            pass
        # intentar configurar también en los módulos cargados (si exponen pytesseract)
        try:
            if hasattr(id_mod, 'pytesseract'):
                try:
                    id_mod.pytesseract.pytesseract.tesseract_cmd = candidate
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if hasattr(qr_mod, 'pytesseract'):
                try:
                    qr_mod.pytesseract.pytesseract.tesseract_cmd = candidate
                except Exception:
                    pass
        except Exception:
            pass
    else:
        # no se encontró tesseract en el sistema; dejar como está (pytesseract usará PATH)
        pass
except Exception:
    # no fallar la importación por problemas al comprobar tesseract
    pass

app = Flask(__name__)


def process_bytes(file_bytes: bytes, filename: str, role: str) -> Dict[str, Any]:
    """Procesa un archivo (PDF o imagen) usando funciones de qr_mod e id_mod.

    role: etiqueta descriptiva ('cedula' o 'rif') usada en la respuesta.
    """
    out: Dict[str, Any] = {'role': role, 'filename': filename, 'type': None, 'qr': [], 'ocr_text': None, 'cedula_refined': None, 'extracted_fields': None, 'validation': None, 'error': None}

    try:
        # No se soportan PDFs en la API; sólo imágenes
        if filename.lower().endswith('.pdf'):
            out['type'] = 'pdf'
            out['error'] = 'pdf files are not supported; please upload an image'
            return out

        out['type'] = 'image'

        # Usar la función única del módulo de QR para procesar bytes
        try:
            qr_result = qr_mod.process_qr_bytes(file_bytes, filename=filename, follow_remote=True)
        except Exception as e:
            out['error'] = f'qr processing failed: {e}'
            return out

        if qr_result and qr_result.get('ok') and qr_result.get('results'):
            out['qr'] = qr_result.get('results')
            # intentar agregar campos extraídos (si el QR apunta a una URL remota y se extrajeron campos)
            extracted = None
            for r in out['qr']:
                if isinstance(r, dict) and r.get('remote') and isinstance(r.get('remote'), dict):
                    remote = r.get('remote')
                    if remote.get('extracted_fields'):
                        extracted = remote.get('extracted_fields')
                        break
            out['extracted_fields'] = extracted
        else:
            # Sin QR: delegar al procesador de cédula (única función del módulo id-validator)
            try:
                id_result = id_mod.process_id_bytes(file_bytes, filename=filename)
            except Exception as e:
                out['error'] = f'id processing failed: {e}'
                return out

            out['ocr_text'] = id_result.get('text')
            out['cedula_refined'] = id_result.get('found_cedula')
            # mantener algunas claves diagnósticas
            if id_result.get('cedula_bbox'):
                out['cedula_diag'] = {
                    'cedula_bbox': id_result.get('cedula_bbox'),
                    'cedula_roi_text': id_result.get('cedula_roi_text'),
                    'cedula_confidence': id_result.get('cedula_confidence')
                }
            out['validation'] = {
                'matched': id_result.get('matched'),
                'match_percentage': id_result.get('match_percentage'),
                'field_scores': id_result.get('field_scores'),
                'record': id_result.get('record'),
                'db_result': id_result.get('db_result')
            }

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

    # Si subieron ambos, agregar comparación entre registro de la DB (cedula) y datos extraídos del RIF
    def _normalize_name(s: str) -> str:
        if not s:
            return ''
        s = s.upper()
        s = re.sub(r"[^A-ZÑÁÉÍÓÚÜ\s]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _digits_only(s: str) -> str:
        if not s:
            return ''
        return re.sub(r"\D", "", str(s))

    def _normalize_rif(s: str) -> str:
        if not s:
            return ''
        s = s.upper()
        s = s.replace(' ', '').replace('\u00A0','')
        # Asegurar formato LETRA-DIGITOS[-DIGITO]
        m = re.match(r"([VEJPG])[-\s]?(\d{1,9})(?:[-\s]?(\d))?", s)
        if m:
            parts = [m.group(1), m.group(2)]
            if m.group(3):
                parts.append(m.group(3))
            return '-'.join(parts)
        # fallback: devolver solo dígitos
        digits = _digits_only(s)
        return digits

    def _similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return difflib.SequenceMatcher(None, a, b).ratio()

    def _compare_record_and_rif(rec: dict, rif_fields: dict) -> dict:
        # rec: registro de la DB normalizado por id-validator; rif_fields: resultado de qr_api
        outc = {'name_similarity': 0.0, 'id_match_ratio': 0.0, 'rif_match_ratio': 0.0, 'overall_match': False, 'details': {}}
        if not rec or not isinstance(rec, dict) or not rif_fields or not isinstance(rif_fields, dict):
            return outc
        # comparar nombres
        rec_name = ' '.join(filter(None, [rec.get('nombres',''), rec.get('apellidos','')]))
        rif_name = rif_fields.get('nombre') or rif_fields.get('name') or rif_fields.get('nombre_completo') or ''
        n1 = _normalize_name(rec_name)
        n2 = _normalize_name(rif_name)
        name_sim = _similarity(n1, n2)
        outc['name_similarity'] = round(name_sim, 3)

        # comparar identificacion / cedula
        rec_ced = rec.get('cedula') or ''
        # rec_ced puede ser 'V-28692795' -> extraer dígitos
        ic1 = _digits_only(rec_ced)
        # rif_fields puede incluir 'id' o 'cedula'
        ic2 = _digits_only(rif_fields.get('identificacion') or rif_fields.get('id') or rif_fields.get('cedula') or rif_fields.get('rif') or '')
        id_ratio = 1.0 if ic1 and ic2 and ic1 == ic2 else (_similarity(ic1, ic2) if (ic1 or ic2) else 0.0)
        outc['id_match_ratio'] = round(id_ratio, 3)

        # comparar rif (numero)
        rec_rif = rif_fields.get('rif') or rif_fields.get('RIF') or ''
        rec_rif_norm = _normalize_rif(rec_rif)
        # también aceptar que el registro tenga cédula como identificación
        rif_ratio = 0.0
        if rec_rif_norm:
            # comparar con la forma normalizada de rec: puede no tener rif en DB
            db_rif = _normalize_rif(rec.get('rif') or '')
            if db_rif:
                rif_ratio = _similarity(db_rif, rec_rif_norm)
            else:
                # comparar digits between rec identification and rif
                rif_ratio = _similarity(_digits_only(rec.get('cedula') or ''), _digits_only(rec_rif_norm))
        outc['rif_match_ratio'] = round(rif_ratio, 3)

        # decidir overall_match: al menos 2 de 3 deben superar umbrales (name>=0.7, id>=0.9 or ==1, rif>=0.9)
        passed = 0
        if name_sim >= 0.7:
            passed += 1
        if id_ratio >= 0.9:
            passed += 1
        if rif_ratio >= 0.9:
            passed += 1
        outc['overall_match'] = passed >= 2
        outc['details'] = {'rec_name': rec_name, 'rif_name': rif_name, 'rec_ced': rec_ced, 'rif_raw': rec_rif}
        return outc

    # si existen ambos resultados, ejecutar comparación
    if 'cedula' in resp['results'] and 'rif' in resp['results']:
        ced_res = resp['results'].get('cedula')
        rif_res = resp['results'].get('rif')
        rec = None
        # preferir record del validation
        if ced_res and ced_res.get('validation') and ced_res['validation'].get('record'):
            rec = ced_res['validation']['record']
        # obtener campos extraídos del RIF
        rif_fields = None
        if rif_res:
            # prefer extracted_fields if present
            rif_fields = rif_res.get('extracted_fields') or (rif_res.get('qr')[0].get('remote').get('extracted_fields') if rif_res.get('qr') and isinstance(rif_res.get('qr')[0], dict) and rif_res.get('qr')[0].get('remote') else None)
        comp = _compare_record_and_rif(rec, rif_fields)
        resp['comparison'] = comp

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

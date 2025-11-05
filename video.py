import re
import cv2
import os
import json
import difflib
import numpy as np
import pytesseract
from flask import Flask, request, jsonify
# ruta al ejecutable de tesseract en Windows (dejar como estaba en tu entorno)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\yrodriguezc\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)


def texto(image: np.ndarray) -> str:
    """Ejecuta OCR sobre la imagen y devuelve (texto).

    Antes la función devolvía un booleano simple; ahora devolvemos sólo el texto y
    delegamos la validación contra la base de datos a otra función.
    """
    # Convertir a gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Filtro (umbral adaptativo como en la versión original)
    umbral = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 25)

    # OCR Configuration
    config = '--psm 1'
    text = pytesseract.image_to_string(umbral, config=config)

    return text


def load_database(path: str):
    """Carga el JSON de base de datos y devuelve una lista de registros."""
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_cedula(s: str) -> str:
    """Normaliza una cédula encontrada a formato 'V-12345678'.

    - Detecta prefijo de nacionalidad (V/E/J/P/G) si existe
    - Elimina separadores en los dígitos (., espacios, comas)
    - Acepta entre 6 y 9 dígitos
    """
    if not s:
        return ''
    s = s.upper()
    # Intentar capturar letra explícita
    letter_match = re.search(r'[VEJPG]', s)
    letter = letter_match.group(0) if letter_match else 'V'
    # Quitar todo lo que no sea dígito
    digits = re.sub(r'\D', '', s)
    if 6 <= len(digits) <= 9:
        return f"{letter}-{digits}"
    return ''


def find_cedula_in_text(text: str) -> str:
    """Intenta extraer la cédula del texto OCR.

    Soporta formatos:
    - V-12345678, V 12.345.678, V12345678
    - 12.345.678 o 12345678 (asumiendo 'V-')
    """
    # 1) Prefijo letra + número con separadores
    m = re.search(r'\b([VEJPG])[-\s]?(\d{1,3}(?:[.,\s]?\d{3})+|\d{6,9})\b', text, flags=re.IGNORECASE)
    if m:
        return normalize_cedula(m.group(0))
    # 2) Sólo número con separadores tipo 12.345.678
    m2 = re.search(r'\b(\d{1,3}(?:[.,\s]?\d{3})+)\b', text)
    if m2:
        return normalize_cedula(m2.group(1))
    # 3) Sólo dígitos largos contiguos
    m3 = re.search(r'\b(\d{6,9})\b', text)
    if m3:
        return normalize_cedula(m3.group(1))
    return ''


def similarity(a: str, b: str) -> float:
    """Devuelve la similitud entre dos strings (0..1) usando SequenceMatcher."""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def compare_text_with_record(text: str, record: dict, fields=None) -> dict:
    """Compara el texto OCR con los campos del registro y devuelve detalle y porcentaje.

    fields: lista de campos a comparar. Si None, usa apellidos, nombres, fecha_nacimiento, nacionalidad.
    """
    if fields is None:
        fields = ['apellidos', 'nombres', 'fecha_nacimiento', 'nacionalidad']

    scores = {}
    total = 0.0
    for f in fields:
        value = record.get(f, '')
        # Para fechas puede ser útil buscar solo año o dígitos
        if f == 'fecha_nacimiento' and value:
            # comparar año si aparece en texto
            year_match = re.search(r'(\d{4})', text)
            if year_match:
                score = 1.0 if year_match.group(1) in value else similarity(value, text)
            else:
                score = similarity(value, text)
        else:
            score = similarity(value, text)
        scores[f] = score
        total += score

    avg = total / len(fields) if fields else 0.0
    # convertir a porcentaje 0..100
    return {'field_scores': scores, 'match_percentage': round(avg * 100, 1)}


def validate_against_db(text: str, db_path: str = 'database_example.json', threshold: float = 15.0) -> dict:
    """Valida el texto OCR contra la base de datos JSON.

    - Extrae la cédula del texto
    - Busca ese registro en el JSON
    - Compara campos y devuelve porcentaje y detalles
    - threshold en porcentaje para considerar match aceptable
    """
    db = load_database(db_path)
    cedula = find_cedula_in_text(text)
    result = {'found_cedula': cedula, 'record': None, 'match_percentage': 0.0, 'field_scores': {}, 'matched': False}
    if not cedula:
        return result

    # Buscar registro en DB: coincidencia por exact match ignorando mayúsculas/espacios
    for rec in db:
        rec_ced = normalize_cedula(rec.get('cedula', ''))
        if rec_ced and rec_ced.replace('-', '') == cedula.replace('-', ''):
            cmp = compare_text_with_record(text, rec)
            result.update({'record': rec, 'match_percentage': cmp['match_percentage'], 'field_scores': cmp['field_scores']})
            # Regla: válido si el porcentaje es mayor o igual a 15%
            # Si prefiere estrictamente por encima (> 15), reemplazar >= por >
            result['matched'] = cmp['match_percentage'] >= threshold
            return result

    return result


@app.route('/validate-text', methods=['GET'])
def validate_text():
    """Endpoint de prueba que permite validar pasando texto directamente (sin imagen).

    Params (query string):
      - text: string con el contenido OCR simulado
      - threshold: porcentaje mínimo para considerar match (por defecto 75)
    """
    text = request.args.get('text', '')
    if not text:
        return jsonify({'error': 'Missing text parameter'}), 400
    try:
        threshold = float(request.args.get('threshold', 15.0))
    except Exception:
        threshold = 15.0

    result = validate_against_db(text, db_path=os.path.join(os.path.dirname(__file__), 'database_example.json'), threshold=threshold)
    message = 'Foto válida.' if result.get('matched') else 'Por favor, tome de nuevo la foto.'
    return jsonify({
        'text': text,
        'found_cedula': result.get('found_cedula'),
        'matched': result.get('matched'),
        'valid': result.get('matched'),
        'match_percentage': result.get('match_percentage'),
        'field_scores': result.get('field_scores'),
        'record': result.get('record') if result.get('record') else None,
        'message': message
    }), 200

@app.route('/')
def index():
    return "Identification OCR API. POST an image to /validate (multipart/form-data, field 'image')."


@app.route('/validate', methods=['POST'])
def validate():
    """Recibe una imagen desde el front-end, ejecuta OCR y devuelve si los datos son válidos.

    Request:
      - multipart/form-data con campo 'image' (archivo)
    Response JSON:
      - valid: bool
      - text: texto extraído (string)
    """
    if 'image' not in request.files:
        return jsonify({'error': "No file part 'image' in request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Leer bytes y decodificar a imagen OpenCV
    data = file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Could not decode image'}), 400

    # Ejecutar OCR
    text = texto(img)

    # Umbral opcional pasado en query/form (porcentaje, por defecto 15)
    try:
        threshold = float(request.form.get('threshold', request.args.get('threshold', 15.0)))
    except Exception:
        threshold = 15.0

    # Validar contra la base de datos local
    result = validate_against_db(text, db_path=os.path.join(os.path.dirname(__file__), 'database_example.json'), threshold=threshold)

    # Respuesta con detalle de matching
    message = 'Foto válida.' if result.get('matched') else 'Por favor, tome de nuevo la foto.'
    return jsonify({
        'text': text,
        'found_cedula': result.get('found_cedula'),
        'matched': result.get('matched'),
        'valid': result.get('matched'),
        'match_percentage': result.get('match_percentage'),
        'field_scores': result.get('field_scores'),
        'record': result.get('record') if result.get('matched') else None,
        'message': message
    }), 200


if __name__ == '__main__':
    # Nota: en producción usar gunicorn/uvicorn u otro servidor de WSGI/ASGI.
    app.run(host='0.0.0.0', port=5000, debug=True)

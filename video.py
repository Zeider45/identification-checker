import re  # Módulo para trabajar con expresiones regulares (búsqueda de patrones de texto)
import cv2  # OpenCV para procesamiento de imágenes
import os  # Operaciones con rutas y el sistema de archivos
import json  # Lectura y escritura de archivos JSON
import difflib  # Cálculo de similitud entre cadenas de texto
import numpy as np  # Manejo de arreglos numéricos (imágenes como matrices)
import pytesseract  # Motor OCR Tesseract desde Python
from pytesseract import Output  # Formato de salida detallado de Tesseract (por palabra)
from flask import Flask, request, jsonify  # Web framework para exponer endpoints HTTP
from uuid import uuid4  # Generación de identificadores únicos (para nombres de archivos)
from datetime import datetime  # Tiempos y fechas (marcar archivos, logs)
import shutil  # Mover/copiar archivos en el sistema de archivos

# ============================
# Constantes y configuración
# ============================

# Ruta al ejecutable de Tesseract en Windows (ajusta según tu equipo)
TESSERACT_CMD = r'C:\Users\yrodriguezc\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'  # Ruta al binario de Tesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD  # Configura Tesseract para pytesseract

# Rutas por defecto y umbrales
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), 'database_example.json')  # Ruta al JSON con datos de prueba
DEFAULT_THRESHOLD = 15.0  # Porcentaje mínimo de coincidencia para considerar "válido"

# Parámetros OCR
FULL_OCR_PSM = '--psm 1'  # Modo de segmentación de página para OCR global
ROI_OCR_PSM = '--psm 7'  # Modo de segmentación de página para OCR en ROI (una línea)
ROI_CHAR_WHITELIST = 'VEJPG0123456789-.,'  # Conjunto de caracteres permitidos para cédula

# Parámetros de binarización adaptativa (preprocesamiento)
ADAPTIVE_BLOCK_SIZE = 55  # Tamaño del bloque para umbral adaptativo
ADAPTIVE_C = 25  # Constante substraída en el umbral adaptativo

# Inicializa la aplicación Flask
app = Flask(__name__)  # Crea la app Flask para exponer endpoints REST


def texto(image: np.ndarray) -> str:
    """Ejecuta OCR sobre la imagen y devuelve el texto detectado.

    Esta función se limita a extraer texto sin validar contra la base de datos.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convierte la imagen de BGR a escala de grises
    umbral = cv2.adaptiveThreshold(  # Aplica umbral adaptativo para resaltar texto
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
    )
    config = FULL_OCR_PSM  # Configuración de Tesseract para OCR global
    text = pytesseract.image_to_string(umbral, config=config)  # Ejecuta OCR y obtiene el texto como string
    return text  # Devuelve el texto extraído


def _extract_cedula_match_span(text: str):
    """Devuelve (match_str, start_idx, end_idx) del primer patrón de cédula encontrado.

    Soporta patrones como "V-12345678", "V 12.345.678", "12345678" y "123-1234567-1".
    Retorna None si no encuentra coincidencias.
    """
    if not text:  # Si no hay texto, no se puede buscar
        return None  # No hay coincidencia
    m = re.search(  # Busca patrón con letra de nacionalidad seguida de número
        r'\b([VEJPG])[-\s]?(\d{1,3}(?:[.,\s]?\d{3})+|\d{6,9})\b', text, flags=re.IGNORECASE
    )
    if m:  # Si encuentra, devuelve el string y posiciones
        return (m.group(0), m.start(), m.end())  # Coincidencia completa y sus índices
    m2 = re.search(r'\b\d{3}-\d{7}-\d\b', text)  # Alternativa: formato dominicano ###-#######-#
    if m2:  # Si encuentra, devuelve datos de la coincidencia
        return (m2.group(0), m2.start(), m2.end())  # Devuelve match y span
    m3 = re.search(r'\b(\d{1,3}(?:[.,\s]?\d{3})+)\b', text)  # Números con separadores (12.345.678)
    if m3:  # Si encuentra agrupación 1 con números
        return (m3.group(1), m3.start(1), m3.end(1))  # Devuelve el grupo numérico y su span
    m4 = re.search(r'\b(\d{6,9})\b', text)  # Sólo dígitos contiguos de longitud 6 a 9
    if m4:  # Si encuentra
        return (m4.group(1), m4.start(1), m4.end(1))  # Devuelve el número y su span
    return None  # Sin coincidencias


def find_cedula_bbox_and_refine(image: np.ndarray, psm_full: str = '--psm 6', psm_roi: str = '--psm 7', pad_ratio: float = 0.15) -> dict:
    """Localiza la cédula en la imagen y mejora el OCR sobre su recorte (ROI).

    Devuelve un dict con:
      - bbox: Caja delimitadora de la cédula {'x','y','w','h'} o None
      - roi_text: Texto OCR sin normalizar del ROI
      - cedula_refined: Cédula normalizada extraída del ROI (si se puede)
      - confidence: Confianza promedio del match en la línea
    """
    result = {  # Estructura de resultado por defecto
        'bbox': None,  # Caja delimitadora de la cédula en la imagen completa
        'roi_text': '',  # Texto OCR extraído del recorte (ROI)
        'cedula_refined': '',  # Cédula normalizada detectada en el ROI
        'confidence': None  # Confianza promedio de tokens relevantes
    }

    if image is None or image.size == 0:  # Verifica que la imagen sea válida
        return result  # Devuelve resultado vacío si no hay imagen

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convierte a escala de grises
    umbral = cv2.adaptiveThreshold(  # Binariza con umbral adaptativo para mejorar OCR
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
    )

    data = pytesseract.image_to_data(  # Pide a Tesseract datos por palabra con coordenadas
        umbral, config=psm_full, output_type=Output.DICT
    )
    n = len(data.get('text', []))  # Número de tokens reconocidos
    if n == 0:  # Si no hay tokens
        return result  # Devuelve resultado vacío

    lines = {}  # Diccionario para agrupar tokens por línea
    for i in range(n):  # Itera por cada token de Tesseract
        txt = (data['text'][i] or '').strip()  # Texto del token, limpiando espacios
        if txt == '' or int(data.get('conf', ['-1'])[i]) == -1:  # Omite tokens vacíos o inválidos
            continue  # Salta a la siguiente iteración
        key = (  # Clave que agrupa tokens en la misma línea (página, bloque, párrafo, línea)
            data.get('page_num', [0])[i],
            data.get('block_num', [0])[i],
            data.get('par_num', [0])[i],
            data.get('line_num', [0])[i]
        )
        lines.setdefault(key, []).append({  # Añade el token a su línea correspondiente
            'text': txt,  # Texto del token
            'left': int(data['left'][i]),  # Coordenada X izquierda del token
            'top': int(data['top'][i]),  # Coordenada Y superior del token
            'width': int(data['width'][i]),  # Ancho del token
            'height': int(data['height'][i]),  # Alto del token
            'conf': float(data.get('conf', ['0'])[i]) if str(data.get('conf', ['0'])[i]).replace('.', '', 1).isdigit() else 0.0  # Confianza
        })

    chosen_bbox = None  # Bbox seleccionada para la cédula
    chosen_conf = None  # Confianza asociada a esa bbox

    for key, tokens in lines.items():  # Recorre cada línea de texto detectada
        if not tokens:  # Si la línea no tiene tokens útiles
            continue  # Pasa a la siguiente línea
        line_text = ''  # Texto completo de la línea
        spans = []  # Lista de spans (inicio, fin, índice de token) para mapear posiciones
        pos = 0  # Posición actual en el string de la línea
        for idx, t in enumerate(tokens):  # Construye el texto y sus spans por token
            token = t['text']  # Texto del token actual
            if idx > 0:  # Si no es el primer token, añade espacio separador
                line_text += ' '  # Inserta un espacio entre tokens
                pos += 1  # Avanza la posición por el espacio
            start = pos  # Marca inicio del token en el texto de línea
            line_text += token  # Agrega el token al texto de la línea
            pos += len(token)  # Avanza posición tantos caracteres como el token
            end = pos  # Fin del token en el texto de línea
            spans.append((start, end, idx))  # Guarda span para mapear a bbox luego

        match = _extract_cedula_match_span(line_text)  # Intenta hallar la cédula en esa línea
        if not match:  # Si no encuentra formato de cédula
            continue  # Pasa a la siguiente línea
        _, m_start, m_end = match  # Obtiene posiciones del match dentro del texto de línea
        idxs = []  # Índices de tokens que caen dentro del match
        for start, end, idx in spans:  # Recorre spans de tokens para cruzarlos con el match
            if not (end <= m_start or start >= m_end):  # Si hay intersección con el match
                idxs.append(idx)  # Añade el índice del token utilizado
        if not idxs:  # Si no hay tokens dentro del match (caso raro)
            continue  # Continúa con la siguiente línea
        xs = [tokens[i]['left'] for i in idxs]  # Lista de X izquierdas de tokens del match
        ys = [tokens[i]['top'] for i in idxs]  # Lista de Y superiores de tokens del match
        xe = [tokens[i]['left'] + tokens[i]['width'] for i in idxs]  # X derechas (left+width)
        ye = [tokens[i]['top'] + tokens[i]['height'] for i in idxs]  # Y inferiores (top+height)
        x1, y1 = min(xs), min(ys)  # Extremo superior-izquierdo de la bbox conjunta
        x2, y2 = max(xe), max(ye)  # Extremo inferior-derecho de la bbox conjunta

        conf_vals = [tokens[i]['conf'] for i in idxs if tokens[i]['conf'] is not None]  # Confianzas de tokens usados
        avg_conf = sum(conf_vals) / len(conf_vals) if conf_vals else None  # Promedio de confianza

        chosen_bbox = (x1, y1, x2 - x1, y2 - y1)  # Guarda bbox como (x, y, w, h)
        chosen_conf = avg_conf  # Guarda confianza asociada
        break  # Rompe el bucle tras la primera coincidencia encontrada

    if not chosen_bbox:  # Si no se encontró ninguna bbox para cédula
        return result  # Devuelve resultado vacío

    x, y, w, h = chosen_bbox  # Desempaqueta la bbox elegida
    pad = int(max(w, h) * pad_ratio)  # Calcula padding proporcional al tamaño de la bbox
    ih, iw = image.shape[:2]  # Alto (ih) y ancho (iw) de la imagen original
    x1 = max(0, x - pad)  # X inicial con padding, limitado a la imagen
    y1 = max(0, y - pad)  # Y inicial con padding, limitado a la imagen
    x2 = min(iw, x + w + pad)  # X final con padding, limitado a la imagen
    y2 = min(ih, y + h + pad)  # Y final con padding, limitado a la imagen

    roi = image[y1:y2, x1:x2]  # Recorta la región de interés (ROI) de la imagen
    if roi.size == 0:  # Si el ROI está vacío (coordenadas degeneradas)
        return result  # Devuelve resultado vacío

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convierte el ROI a escala de grises
    roi_thr = cv2.adaptiveThreshold(  # Binariza el ROI para mejorar reconocimiento
        roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
    )
    roi_config = f"{psm_roi} -c tessedit_char_whitelist={ROI_CHAR_WHITELIST}"  # Limita caracteres válidos en ROI
    roi_text = pytesseract.image_to_string(roi_thr, config=roi_config)  # OCR sobre el ROI

    result['bbox'] = {'x': int(x1), 'y': int(y1), 'w': int(x2 - x1), 'h': int(y2 - y1)}  # Guarda bbox del ROI (con padding)
    result['roi_text'] = roi_text  # Texto crudo obtenido del ROI
    result['cedula_refined'] = normalize_cedula(find_cedula_in_text(roi_text)) or normalize_cedula(roi_text)  # Cédula refinada
    result['confidence'] = chosen_conf  # Confianza promedio de tokens en la línea
    return result  # Devuelve el resultado completo


def _canonicalize_record(rec: dict) -> dict:
    """Convierte un registro de distintos formatos al formato canónico que usa la app.

    Formato canónico esperado por el resto del código:
      - cedula: "V-12345678"
      - nacionalidad: "V" | "E" | ...
      - nombres: "NOMBRE [SEGUNDO]"
      - apellidos: "APELLIDO [SEGUNDO]"
      - fecha_nacimiento: "YYYY-MM-DD" (si está disponible)
    """
    if not isinstance(rec, dict):
        return {}

    # Detecta el nuevo formato (ej.: claves: intcedula, strnacionalidad, dtmacimiento, etc.)
    if 'intcedula' in rec or 'strnacionalidad' in rec:
        nationality = str(rec.get('strnacionalidad', '') or '').strip().upper() or 'V'
        # Asegura extraer sólo dígitos (por si viene como string con separadores)
        ced_num_raw = rec.get('intcedula', '')
        ced_num = ''.join(ch for ch in str(ced_num_raw) if ch.isdigit())
        cedula = f"{nationality}-{ced_num}" if ced_num else ''

        nombres = ' '.join([
            str(rec.get('strnombre_primer', '') or '').strip(),
            str(rec.get('strnombre_segundo', '') or '').strip()
        ]).strip()
        apellidos = ' '.join([
            str(rec.get('strapellido_primer', '') or '').strip(),
            str(rec.get('strapellido_segundo', '') or '').strip()
        ]).strip()

        # Fecha: tomar sólo la parte de fecha si viene con tiempo (YYYY-MM-DDThh:mm:ss)
        fecha_raw = str(rec.get('dtmacimiento', '') or '').strip()
        fecha_nacimiento = ''
        if fecha_raw:
            fecha_nacimiento = fecha_raw.split('T', 1)[0]

        return {
            'cedula': cedula,
            'nacionalidad': nationality,
            'nombres': nombres,
            'apellidos': apellidos,
            'fecha_nacimiento': fecha_nacimiento,
            # Campos adicionales por si se quieren usar en el futuro
            'estado_civil': rec.get('strestado_civil')
        }

    # Si ya viene en formato antiguo/canónico, devuelve tal cual (haciendo copias seguras)
    return {
        'cedula': str(rec.get('cedula', '') or ''),
        'nacionalidad': str(rec.get('nacionalidad', '') or ''),
        'nombres': str(rec.get('nombres', '') or ''),
        'apellidos': str(rec.get('apellidos', '') or ''),
        'fecha_nacimiento': str(rec.get('fecha_nacimiento', '') or ''),
        'estado_civil': rec.get('estado_civil')
    }


def load_database(path: str):
    """Carga el JSON y lo adapta al formato canónico que usa la app."""
    if not os.path.exists(path):  # Verifica que el archivo exista
        return []  # Si no existe, devuelve lista vacía
    with open(path, 'r', encoding='utf-8') as f:  # Abre el archivo en modo lectura UTF-8
        data = json.load(f)  # Parsea el JSON
    # Adapta listas o dicts individuales
    if isinstance(data, list):
        return [_canonicalize_record(rec) for rec in data]
    if isinstance(data, dict):
        # Si el JSON es un objeto único, regresa lista de uno para mantener contrato
        return [_canonicalize_record(data)]
    return []


def normalize_cedula(s: str) -> str:
    """Normaliza una cédula encontrada a formato 'V-12345678'.

    - Detecta prefijo de nacionalidad (V/E/J/P/G) si existe
    - Elimina separadores en los dígitos (., espacios, comas)
    - Acepta entre 6 y 9 dígitos
    """
    if not s:  # Si el string es vacío o None
        return ''  # No hay cédula que normalizar
    s = s.upper()  # Convierte a mayúsculas para unificar formatos
    letter_match = re.search(r'[VEJPG]', s)  # Busca una letra de nacionalidad explícita
    letter = letter_match.group(0) if letter_match else 'V'  # Usa la letra encontrada o asume 'V'
    digits = re.sub(r'\D', '', s)  # Elimina todos los caracteres que no sean dígitos
    if 6 <= len(digits) <= 9:  # Valida longitud típica de cédula
        return f"{letter}-{digits}"  # Retorna con formato estándar LETRA-########
    return ''  # Si la longitud no cuadra, devuelve vacío


def find_cedula_in_text(text: str) -> str:
    """Intenta extraer la cédula del texto OCR.

    Soporta formatos:
    - V-12345678, V 12.345.678, V12345678
    - 12.345.678 o 12345678 (asumiendo 'V-')
    """
    m = re.search(  # Prefijo de letra + número con separadores o continuo
        r'\b([VEJPG])[-\s]?(\d{1,3}(?:[.,\s]?\d{3})+|\d{6,9})\b', text, flags=re.IGNORECASE
    )
    if m:  # Si hay coincidencia con letra
        return normalize_cedula(m.group(0))  # Normaliza y devuelve
    m2 = re.search(r'\b(\d{1,3}(?:[.,\s]?\d{3})+)\b', text)  # Números con separadores (sin letra)
    if m2:  # Si encuentra
        return normalize_cedula(m2.group(1))  # Normaliza como V- por defecto
    m3 = re.search(r'\b(\d{6,9})\b', text)  # Sólo dígitos contiguos sin letra
    if m3:  # Si encuentra
        return normalize_cedula(m3.group(1))  # Normaliza como V- por defecto
    return ''  # No se halló cédula en el texto


def similarity(a: str, b: str) -> float:
    """Devuelve la similitud entre dos strings (0..1) usando SequenceMatcher."""
    if not a or not b:  # Si alguno está vacío
        return 0.0  # Similitud nula
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()  # Ratio de similitud [0,1]


def compare_text_with_record(text: str, record: dict, fields=None) -> dict:
    """Compara el texto OCR con los campos del registro y devuelve detalle y porcentaje.

    fields: lista de campos a comparar. Si None, usa apellidos, nombres, fecha_nacimiento, nacionalidad.
    """
    if fields is None:  # Si no se especifican campos para comparar
        fields = ['apellidos', 'nombres', 'fecha_nacimiento', 'nacionalidad']  # Campos por defecto

    scores = {}  # Diccionario de puntuaciones por campo
    total = 0.0  # Acumulador de similitudes
    for f in fields:  # Recorre cada campo a evaluar
        value = record.get(f, '')  # Obtiene el valor del registro para ese campo
        if f == 'fecha_nacimiento' and value:  # Si es fecha de nacimiento
            year_match = re.search(r'(\d{4})', text)  # Busca un año (4 dígitos) en el texto OCR
            if year_match:  # Si aparece un año en el texto
                score = 1.0 if year_match.group(1) in value else similarity(value, text)  # Puntúa 1.0 si el año coincide, si no usa similitud
            else:  # Si no hay año explícito en el texto
                score = similarity(value, text)  # Usa similitud general de cadenas
        else:  # Para otros campos (apellidos, nombres, nacionalidad)
            score = similarity(value, text)  # Calcula similitud de strings
        scores[f] = score  # Guarda el score del campo
        total += score  # Suma al total

    avg = total / len(fields) if fields else 0.0  # Promedia las similitudes
    return {'field_scores': scores, 'match_percentage': round(avg * 100, 1)}  # Devuelve detalle y porcentaje


def validate_against_db(text: str, db_path: str = 'database_example.json', threshold: float = 15.0, cedula_hint: str | None = None) -> dict:
    """Valida el texto OCR contra la base de datos JSON.

    - Extrae la cédula del texto
    - Busca ese registro en el JSON
    - Compara campos y devuelve porcentaje y detalles
    - threshold en porcentaje para considerar match aceptable
    """
    db = load_database(db_path)  # Carga la base de datos desde el JSON
    cedula = cedula_hint or find_cedula_in_text(text)  # Usa cédula refinada del ROI o extráela del texto global
    result = {  # Arma la estructura básica del resultado
        'found_cedula': cedula,  # Cédula detectada (normalizada)
        'record': None,  # Registro encontrado en DB (si existe)
        'match_percentage': 0.0,  # Porcentaje de coincidencia acumulado
        'field_scores': {},  # Puntajes por campo comparado
        'matched': False  # Indicador de si supera el umbral
    }
    if not cedula:  # Si no hay cédula, no puede validarse contra la base de datos
        return result  # Devuelve el resultado sin coincidencia

    for rec in db:  # Recorre los registros de la base de datos
        rec_ced = normalize_cedula(rec.get('cedula', ''))  # Normaliza la cédula del registro
        if rec_ced and rec_ced.replace('-', '') == cedula.replace('-', ''):  # Compara ignorando el guion
            cmp = compare_text_with_record(text, rec)  # Calcula similitudes campo a campo
            result.update({  # Actualiza resultado con comparaciones
                'record': rec,
                'match_percentage': cmp['match_percentage'],
                'field_scores': cmp['field_scores']
            })
            result['matched'] = cmp['match_percentage'] >= threshold  # Cumple si supera o iguala el umbral
            return result  # Devuelve inmediatamente tras la primera coincidencia por cédula

    return result  # Si ninguna cédula coincide, retorna sin match



@app.route('/')
def index():
    return "Identification OCR API. POST an image to /validate (multipart/form-data, field 'image')."  # Mensaje básico en la raíz


@app.route('/validate', methods=['POST'])
def validate():
    """Recibe una imagen desde el front-end, ejecuta OCR y devuelve si los datos son válidos.

    Request:
      - multipart/form-data con campo 'image' (archivo)
    Response JSON:
      - valid: bool
      - text: texto extraído (string)
    """
    if 'image' not in request.files:  # Verifica que se haya enviado el campo 'image'
        return jsonify({'error': "No file part 'image' in request"}), 400  # Error 400 si falta el archivo

    file = request.files['image']  # Obtiene el archivo del request
    if file.filename == '':  # Si el nombre de archivo viene vacío
        return jsonify({'error': 'No selected file'}), 400  # Error 400 por archivo no seleccionado

    data = file.read()  # Lee los bytes del archivo recibido
    nparr = np.frombuffer(data, np.uint8)  # Crea un arreglo numpy desde el buffer de bytes
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decodifica los bytes a imagen OpenCV en color
    if img is None:  # Si la decodificación falla
        return jsonify({'error': 'Could not decode image'}), 400  # Error 400 por imagen inválida

    text = texto(img)  # Ejecuta OCR global y obtiene el texto completo
    cedula_loc = find_cedula_bbox_and_refine(img)  # Intenta localizar la cédula y refinar su OCR por ROI
    cedula_hint = cedula_loc.get('cedula_refined') if cedula_loc else None  # Sugiere cédula detectada para la validación

    try:  # Intenta obtener el umbral desde form o query
        threshold = float(request.form.get('threshold', request.args.get('threshold', DEFAULT_THRESHOLD)))  # Umbral de coincidencia
    except Exception:  # Si falla, usa el predeterminado
        threshold = DEFAULT_THRESHOLD  # Umbral por defecto

    result = validate_against_db(  # Valida el texto contra la base de datos local
        text, db_path=DEFAULT_DB_PATH, threshold=threshold, cedula_hint=cedula_hint
    )

    # (Se eliminó el guardado de recortes en la carpeta 'recortes')

    # Si la validación fue exitosa, mover/guardar la imagen en "cedulas-aprobadas"
    moved_path = None
    if result.get('matched'):
        base_dir = os.path.dirname(__file__)
        aprobadas_dir = os.path.join(base_dir, 'cedulas-aprobadas')
        os.makedirs(aprobadas_dir, exist_ok=True)

        # Intentar mover desde la carpeta "cedulas" si existe un archivo con el mismo nombre
        original_name = (file.filename or '').strip()
        safe_name = os.path.basename(original_name) or ''
        src_on_disk = os.path.join(base_dir, 'cedulas', safe_name) if safe_name else None
        # Determinar la cédula en formato dígitos (nombre del archivo = número de cédula)
        ced_norm = (cedula_loc.get('cedula_refined') if cedula_loc else '') or (result.get('found_cedula') or '')
        ced_digits = re.sub(r'\D', '', ced_norm or '')
        if not ced_digits and result.get('record'):
            rec_ced = str(result['record'].get('cedula', '') or '')
            ced_digits = re.sub(r'\D', '', rec_ced)
        # Fallback mínimo si no se pudo extraer número
        if not ced_digits:
            ced_digits = 'cedula'

        # Determinar extensión preferida
        _, ext = os.path.splitext(safe_name)
        if not ext:
            ext = '.png'

        # Evitar sobrescrituras: si existe, agregar sufijo incremental
        base_name = ced_digits
        candidate = f"{base_name}{ext}"
        dest_path = os.path.join(aprobadas_dir, candidate)
        if os.path.exists(dest_path):
            i = 1
            while True:
                candidate = f"{base_name}-{i}{ext}"
                dest_path = os.path.join(aprobadas_dir, candidate)
                if not os.path.exists(dest_path):
                    break
                i += 1

        try:
            if src_on_disk and os.path.exists(src_on_disk):
                # Mover el archivo existente desde "cedulas" hacia "cedulas-aprobadas"
                shutil.move(src_on_disk, dest_path)
                moved_path = dest_path
            else:
                # Guardar la imagen recibida (bytes del request) en "cedulas-aprobadas"
                # Usamos OpenCV para asegurar formato correcto
                ok = False
                # Si la extensión es .png/.jpg, respetar; por robustez usamos imwrite con la imagen decodificada
                try:
                    ok = cv2.imwrite(dest_path, img)
                except Exception:
                    ok = False
                if ok:
                    moved_path = dest_path
        except Exception as e:
            # No interrumpir la respuesta si falla el movimiento; reportar más abajo
            moved_path = None

    message = 'Foto válida.' if result.get('matched') else 'Por favor, tome de nuevo la foto.'  # Mensaje según validación
    resp = {  # Construye la respuesta principal del endpoint
        'text': text,  # Texto OCR global
        'found_cedula': result.get('found_cedula'),  # Cédula detectada/fundamentada
        'matched': result.get('matched'),  # Indicador booleano de match
        'valid': result.get('matched'),  # Alias de matched
        'match_percentage': result.get('match_percentage'),  # Porcentaje de coincidencia
        'field_scores': result.get('field_scores'),  # Detalle por campo
        'record': result.get('record') if result.get('matched') else None,  # Registro solo si hay match
        'message': message  # Mensaje de ayuda para la UI
    }
    if moved_path:
        resp['approved_saved_path'] = moved_path  # Ruta del archivo aprobado movido/guardado
    if cedula_loc and cedula_loc.get('bbox'):  # Si se detectó ROI de cédula, agrega datos diagnósticos
        resp['cedula_bbox'] = cedula_loc['bbox']  # Bbox del ROI con padding
        resp['cedula_roi_text'] = cedula_loc.get('roi_text')  # Texto OCR del ROI
        resp['cedula_refined'] = cedula_loc.get('cedula_refined')  # Cédula refinada y normalizada
        if cedula_loc.get('confidence') is not None:  # Si hay confianza calculada
            resp['cedula_confidence'] = round(float(cedula_loc['confidence']), 2)  # Redondea la confianza

    return jsonify(resp), 200  # Devuelve la respuesta JSON con HTTP 200 OK


if __name__ == '__main__':
    # Nota: en producción usar gunicorn/uvicorn u otro servidor de WSGI/ASGI.
    app.run(host='0.0.0.0', port=5000, debug=True)  # Arranca el servidor de desarrollo Flask

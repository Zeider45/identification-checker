import re
import os
import json
import difflib
import numpy as np
import cv2
import pytesseract
from pytesseract import Output
import shutil


def process_id_bytes(data: bytes, filename: str = None, db_path: str = None, threshold: float = None, save_approved: bool = True) -> dict:
    """Procesa bytes de una imagen de cédula.

    Esta función agrupa internamente las rutinas de OCR, búsqueda de cédula,
    validación contra la base de datos y (opcional) guardado de imágenes
    aprobadas. Devuelve un dict con información similar a la API previa.

    Todos los helpers están definidos dentro de esta función para que el
    módulo exponga únicamente esta función a nivel top-level.
    """

    # --- constantes y parámetros locales (antes estaban a nivel módulo)
    ADAPTIVE_BLOCK_SIZE = 55
    ADAPTIVE_C = 25
    FULL_OCR_PSM = '--psm 1'
    ROI_OCR_PSM = '--psm 7'
    ROI_CHAR_WHITELIST = 'VEJPG0123456789-.,'
    DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), 'database_example.json')
    DEFAULT_THRESHOLD = 15.0

    # --- helpers anidados
    def _normalize_cedula(s: str) -> str:
        if not s:
            return ''
        ss = s.upper()
        letter_match = re.search(r'[VEJPG]', ss)
        letter = letter_match.group(0) if letter_match else 'V'
        digits = re.sub(r'\D', '', ss)
        if 6 <= len(digits) <= 9:
            return f"{letter}-{digits}"
        return ''

    def _find_cedula_in_text(text: str) -> str:
        if not text:
            return ''
        m = re.search(r'\b([VEJPG])[-\s]?(\d{1,3}(?:[.,\s]?\d{3})+|\d{6,9})\b', text, flags=re.IGNORECASE)
        if m:
            return _normalize_cedula(m.group(0))
        m2 = re.search(r'\b(\d{1,3}(?:[.,\s]?\d{3})+)\b', text)
        if m2:
            return _normalize_cedula(m2.group(1))
        m3 = re.search(r'\b(\d{6,9})\b', text)
        if m3:
            return _normalize_cedula(m3.group(1))
        return ''

    def _similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _compare_text_with_record(text: str, record: dict, fields=None) -> dict:
        if fields is None:
            fields = ['apellidos', 'nombres', 'fecha_nacimiento', 'nacionalidad']
        scores = {}
        total = 0.0
        for f in fields:
            value = record.get(f, '')
            if f == 'fecha_nacimiento' and value:
                year_match = re.search(r'(\d{4})', text)
                if year_match:
                    score = 1.0 if year_match.group(1) in value else _similarity(value, text)
                else:
                    score = _similarity(value, text)
            else:
                score = _similarity(value, text)
            scores[f] = score
            total += score
        avg = total / len(fields) if fields else 0.0
        return {'field_scores': scores, 'match_percentage': round(avg * 100, 1)}

    def _load_database(path: str):
        if not path:
            path = DEFAULT_DB_PATH
        if not os.path.exists(path):
            return []
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        out = []
        def _map_record(rec: dict) -> dict:
            # Mapea distintos esquemas posibles a nuestro formato estándar
            # Priorizar campos ya en formato esperado
            ced = ''
            nat = ''
            nombres = ''
            apellidos = ''
            fecha = ''
            estado = None

            # formato "nuevo" (esperado)
            if 'cedula' in rec or 'nacionalidad' in rec or 'nombres' in rec or 'apellidos' in rec:
                ced = str(rec.get('cedula', '') or '')
                nat = str(rec.get('nacionalidad', '') or '')
                nombres = str(rec.get('nombres', '') or '')
                apellidos = str(rec.get('apellidos', '') or '')
                fecha = str(rec.get('fecha_nacimiento', '') or '')
                estado = rec.get('estado_civil')
                return {'cedula': ced, 'nacionalidad': nat, 'nombres': nombres, 'apellidos': apellidos, 'fecha_nacimiento': fecha, 'estado_civil': estado}

            # detectar esquema del `database_example.json` (campos con prefijo str/int)
            if 'intcedula' in rec or 'strnacionalidad' in rec:
                try:
                    ced_digits = str(int(rec.get('intcedula'))) if rec.get('intcedula') is not None else ''
                except Exception:
                    ced_digits = str(rec.get('intcedula') or '')
                nat = str(rec.get('strnacionalidad', '') or '')
                ced = f"{nat}-{ced_digits}" if ced_digits else ''
                # concatenar nombres y apellidos
                nombre_parts = []
                if rec.get('strnombre_primer'):
                    nombre_parts.append(str(rec.get('strnombre_primer')))
                if rec.get('strnombre_segundo'):
                    nombre_parts.append(str(rec.get('strnombre_segundo')))
                nombres = ' '.join(nombre_parts).strip()
                apellido_parts = []
                if rec.get('strapellido_primer'):
                    apellido_parts.append(str(rec.get('strapellido_primer')))
                if rec.get('strapellido_segundo'):
                    apellido_parts.append(str(rec.get('strapellido_segundo')))
                apellidos = ' '.join(apellido_parts).strip()
                # fecha: varios nombres posibles (dtmacimiento es el presente en el ejemplo)
                fecha = str(rec.get('dtmacimiento') or rec.get('dt_nacimiento') or rec.get('fecha_nacimiento') or '')
                estado = rec.get('strestado_civil') or rec.get('estado_civil') or rec.get('cintestado_civil')
                return {'cedula': ced, 'nacionalidad': nat, 'nombres': nombres, 'apellidos': apellidos, 'fecha_nacimiento': fecha, 'estado_civil': estado}

            # fallback: intentar mapear cualquier clave parecida
            ced = str(rec.get('cedula', '') or rec.get('id', '') or '')
            nombres = str(rec.get('nombres', '') or rec.get('nombre', '') or '')
            apellidos = str(rec.get('apellidos', '') or rec.get('apellido', '') or '')
            fecha = str(rec.get('fecha_nacimiento', '') or '')
            estado = rec.get('estado_civil')
            return {'cedula': ced, 'nacionalidad': nat, 'nombres': nombres, 'apellidos': apellidos, 'fecha_nacimiento': fecha, 'estado_civil': estado}

        if isinstance(data, list):
            for rec in data:
                if isinstance(rec, dict):
                    out.append(_map_record(rec))
        elif isinstance(data, dict):
            out.append(_map_record(data))
        return out

    def _texto_from_image(img: np.ndarray) -> str:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        umbral = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C)
        text = pytesseract.image_to_string(umbral, config=FULL_OCR_PSM)
        return text

    def _find_cedula_bbox_and_refine_local(image: np.ndarray, psm_full: str = '--psm 6', psm_roi: str = '--psm 7', pad_ratio: float = 0.15) -> dict:
        result = {'bbox': None, 'roi_text': '', 'cedula_refined': '', 'confidence': None}
        if image is None or image.size == 0:
            return result
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        umbral = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C)
        data = pytesseract.image_to_data(umbral, config=psm_full, output_type=Output.DICT)
        n = len(data.get('text', []))
        if n == 0:
            return result
        lines = {}
        for i in range(n):
            txt = (data['text'][i] or '').strip()
            if txt == '' or int(data.get('conf', ['-1'])[i]) == -1:
                continue
            key = (data.get('page_num', [0])[i], data.get('block_num', [0])[i], data.get('par_num', [0])[i], data.get('line_num', [0])[i])
            lines.setdefault(key, []).append({
                'text': txt,
                'left': int(data['left'][i]),
                'top': int(data['top'][i]),
                'width': int(data['width'][i]),
                'height': int(data['height'][i]),
                'conf': float(data.get('conf', ['0'])[i]) if str(data.get('conf', ['0'])[i]).replace('.', '', 1).isdigit() else 0.0
            })

        chosen_bbox = None
        chosen_conf = None
        for key, tokens in lines.items():
            if not tokens:
                continue
            line_text = ''
            spans = []
            pos = 0
            for idx, t in enumerate(tokens):
                token = t['text']
                if idx > 0:
                    line_text += ' '
                    pos += 1
                start = pos
                line_text += token
                pos += len(token)
                end = pos
                spans.append((start, end, idx))
            # buscar cédula en la línea
            m = re.search(r'\b([VEJPG])[-\s]?(\d{1,3}(?:[.,\s]?\d{3})+|\d{6,9})\b', line_text, flags=re.IGNORECASE)
            if not m:
                m2 = re.search(r'\b(\d{1,3}(?:[.,\s]?\d{3})+)\b', line_text)
                m = m2
            if not m:
                m3 = re.search(r'\b(\d{6,9})\b', line_text)
                if m3:
                    m = m3
            if not m:
                continue
            m_start, m_end = (m.start(), m.end()) if m.groups() is None else (m.start(), m.end())
            idxs = []
            for start, end, idx in spans:
                if not (end <= m_start or start >= m_end):
                    idxs.append(idx)
            if not idxs:
                continue
            xs = [tokens[i]['left'] for i in idxs]
            ys = [tokens[i]['top'] for i in idxs]
            xe = [tokens[i]['left'] + tokens[i]['width'] for i in idxs]
            ye = [tokens[i]['top'] + tokens[i]['height'] for i in idxs]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xe), max(ye)
            conf_vals = [tokens[i]['conf'] for i in idxs if tokens[i]['conf'] is not None]
            avg_conf = sum(conf_vals) / len(conf_vals) if conf_vals else None
            chosen_bbox = (x1, y1, x2 - x1, y2 - y1)
            chosen_conf = avg_conf
            break

        if not chosen_bbox:
            return result
        x, y, w, h = chosen_bbox
        pad = int(max(w, h) * pad_ratio)
        ih, iw = image.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(iw, x + w + pad)
        y2 = min(ih, y + h + pad)
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return result
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_thr = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C)
        roi_config = f"{psm_roi} -c tessedit_char_whitelist={ROI_CHAR_WHITELIST}"
        roi_text = pytesseract.image_to_string(roi_thr, config=roi_config)
        result['bbox'] = {'x': int(x1), 'y': int(y1), 'w': int(x2 - x1), 'h': int(y2 - y1)}
        result['roi_text'] = roi_text
        result['cedula_refined'] = _normalize_cedula(_find_cedula_in_text(roi_text)) or _normalize_cedula(roi_text)
        result['confidence'] = chosen_conf
        return result

    def _validate_against_db(text: str, db_path_local: str = None, threshold_local: float = None, cedula_hint_local: str | None = None) -> dict:
        if db_path_local is None:
            db_path_local = DEFAULT_DB_PATH
        if threshold_local is None:
            threshold_local = DEFAULT_THRESHOLD
        db = _load_database(db_path_local)
        cedula = cedula_hint_local or _find_cedula_in_text(text)
        result = {'found_cedula': cedula, 'record': None, 'match_percentage': 0.0, 'field_scores': {}, 'matched': False}
        if not cedula:
            return result
        for rec in db:
            rec_ced = _normalize_cedula(rec.get('cedula', ''))
            if rec_ced and rec_ced.replace('-', '') == cedula.replace('-', ''):
                cmp = _compare_text_with_record(text, rec)
                result.update({'record': rec, 'match_percentage': cmp['match_percentage'], 'field_scores': cmp['field_scores']})
                result['matched'] = cmp['match_percentage'] >= threshold_local
                return result
        return result

    # --- inicio del procesamiento principal
    out: dict = {'text': None, 'found_cedula': None, 'matched': False, 'match_percentage': 0.0, 'field_scores': {}, 'record': None, 'message': None}
    # decodificar imagen
    try:
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        img = None
    if img is None:
        out['message'] = 'invalid image'
        return out

    # OCR global
    try:
        text = _texto_from_image(img)
    except Exception:
        text = ''
    out['text'] = text

    # intentar localizar cedula y refinar
    try:
        ced_loc = _find_cedula_bbox_and_refine_local(img)
        ced_refined = ced_loc.get('cedula_refined') if isinstance(ced_loc, dict) else None
    except Exception:
        ced_loc = None
        ced_refined = None

    dbp = db_path or DEFAULT_DB_PATH
    thr = threshold if threshold is not None else DEFAULT_THRESHOLD
    result = _validate_against_db(text, db_path_local=dbp, threshold_local=thr, cedula_hint_local=ced_refined)
    # Incluir siempre los datos de comparación en la respuesta para diagnóstico.
    # Antes se ocultaba el 'record' cuando no se alcanzaba el umbral; eso dificultaba
    # entender por qué no se consideraba una coincidencia. Ahora devolvemos el registro
    # (si existe) y el resultado bruto de la validación en `db_result`.
    out.update({
        'found_cedula': result.get('found_cedula'),
        'matched': result.get('matched'),
        'match_percentage': result.get('match_percentage'),
        'field_scores': result.get('field_scores'),
        'record': result.get('record'),
        'db_result': result
    })

    # guardar imagen aprobada si corresponde
    moved_path = None
    if out.get('matched') and save_approved:
        base_dir = os.path.dirname(__file__)
        aprobadas_dir = os.path.join(base_dir, 'cedulas-aprobadas')
        os.makedirs(aprobadas_dir, exist_ok=True)
        safe_name = os.path.basename(filename or '')
        ced_norm = (ced_refined or '') or (result.get('found_cedula') or '')
        ced_digits = re.sub(r'\D', '', ced_norm or '')
        if not ced_digits and result.get('record'):
            rec_ced = str(result['record'].get('cedula', '') or '')
            ced_digits = re.sub(r'\D', '', rec_ced)
        if not ced_digits:
            ced_digits = 'cedula'
        _, ext = os.path.splitext(safe_name)
        if not ext:
            ext = '.png'
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
            ok = False
            try:
                ok = cv2.imwrite(dest_path, img)
            except Exception:
                ok = False
            if ok:
                moved_path = dest_path
        except Exception:
            moved_path = None

    if moved_path:
        out['approved_saved_path'] = moved_path
    if ced_loc and ced_loc.get('bbox'):
        out['cedula_bbox'] = ced_loc['bbox']
        out['cedula_roi_text'] = ced_loc.get('roi_text')
        out['cedula_refined'] = ced_loc.get('cedula_refined')
        if ced_loc.get('confidence') is not None:
            out['cedula_confidence'] = round(float(ced_loc['confidence']), 2)

    out['message'] = 'Foto válida.' if out.get('matched') else 'Por favor, tome de nuevo la foto.'
    return out
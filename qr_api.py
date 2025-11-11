import io
# importar el módulo sys para leer argumentos y controlar el arranque
import sys
# importar tipos para anotaciones: listas, diccionarios, Any, Optional, Set y Tuple
from typing import List, Dict, Any, Optional, Set, Tuple
import re
import html as _html_lib

# Flask: framework web ligero
from flask import Flask, request, jsonify
import io
import sys
from typing import List, Dict, Any, Optional
import re
import html as _html_lib
import cv2
import numpy as np
import requests
from PIL import Image
import pytesseract


def process_qr_bytes(data: bytes, filename: str = None, follow_remote: bool = True, max_size: int = 20 * 1024 * 1024) -> Dict[str, Any]:
    """Procesa bytes de una imagen para detectar QRs y (opcionalmente) seguir URLs.

    Devuelve un diccionario similar a la API previa: {'ok': bool, 'error': str|None, 'results': [...]}
    El resto de helpers están anidados para que el módulo solo exponga esta función.
    """

    def _image_from_bytes_local(data_bytes: bytes) -> Optional[np.ndarray]:
        try:
            arr = np.frombuffer(data_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None

    def _detect_qr_local(img: np.ndarray) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if img is None:
            return results

        def try_opencv(image: np.ndarray) -> List[Dict[str, Any]]:
            out = []
            qr = cv2.QRCodeDetector()
            if hasattr(qr, 'detectAndDecodeMulti'):
                try:
                    ok, decoded_info, points, _ = qr.detectAndDecodeMulti(image)
                except Exception:
                    ok = False
                    decoded_info = []
                    points = None
                if ok and decoded_info:
                    for idx, text in enumerate(decoded_info):
                        if not text:
                            continue
                        pts = None
                        try:
                            if points is not None and len(points) > idx:
                                pts = points[idx].tolist()
                        except Exception:
                            pts = None
                        out.append({'data': text, 'points': pts})
                    return out
            try:
                text, pts = qr.detectAndDecode(image)
                if text:
                    pts_list = None
                    if pts is not None and getattr(pts, 'size', 0):
                        pts_list = pts.tolist()
                    out.append({'data': text, 'points': pts_list})
            except Exception:
                pass
            return out

        def try_pyzbar(image: np.ndarray) -> List[Dict[str, Any]]:
            out = []
            try:
                from pyzbar import pyzbar
            except Exception:
                return out
            decs = pyzbar.decode(image)
            for d in decs:
                try:
                    pts = [(p.x, p.y) for p in d.polygon] if d.polygon else None
                except Exception:
                    pts = None
                try:
                    text = d.data.decode('utf-8')
                except Exception:
                    text = d.data.decode(errors='ignore')
                out.append({'data': text, 'points': pts})
            return out

        def generate_variants(image: np.ndarray) -> List[np.ndarray]:
            variants = [image]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            variants.append(gray)
            try:
                eq = cv2.equalizeHist(gray)
                variants.append(eq)
            except Exception:
                pass
            try:
                at = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8)
                variants.append(at)
            except Exception:
                pass
            try:
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                variants.append(th)
            except Exception:
                pass
            h, w = gray.shape[:2]
            for scale in (1.5, 2.0):
                try:
                    nw, nh = int(w * scale), int(h * scale)
                    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
                    variants.append(resized)
                except Exception:
                    pass
            angles = (-15, 15, -7, 7)
            for a in angles:
                try:
                    M = cv2.getRotationMatrix2D((w // 2, h // 2), a, 1.0)
                    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                    variants.append(rotated)
                except Exception:
                    pass
            try:
                variants.append(cv2.flip(image, 1))
                variants.append(cv2.flip(image, 0))
            except Exception:
                pass
            uniq = []
            seen = set()
            for v in variants:
                key = (v.shape[0], v.shape[1], getattr(v, 'ndim', 2))
                if key not in seen:
                    seen.add(key)
                    uniq.append(v)
            return uniq

        seen_texts = set()
        variants = generate_variants(img)
        for var in variants:
            try:
                op_results = try_opencv(var)
            except Exception:
                op_results = []
            for r in op_results:
                txt = r.get('data')
                if txt and txt not in seen_texts:
                    seen_texts.add(txt)
                    results.append(r)
            try:
                p_results = try_pyzbar(var)
            except Exception:
                p_results = []
            for r in p_results:
                txt = r.get('data')
                if txt and txt not in seen_texts:
                    seen_texts.add(txt)
                    results.append(r)
            if results:
                break
        return results

    def _parse_fields_from_text_local(text: str) -> Dict[str, Any]:
        if not text:
            return {"dates": [], "name": None, "address": None, "rif": None}
        txt = text.replace('\r', '\n')
        txt_norm = '\n'.join(line.strip() for line in txt.splitlines() if line.strip())
        dates = []
        date_patterns = [r"\b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}\b", r"\b\d{4}-\d{2}-\d{2}\b"]
        for pat in date_patterns:
            for m in re.findall(pat, txt_norm):
                if m not in dates:
                    dates.append(m)
        rif = None
        m = re.search(r"rif[:\s]*([VEJPG\-\s]?\d{6,9}-?\d?)", txt_norm, flags=re.IGNORECASE)
        if m:
            rif_raw = m.group(1)
            rif = re.sub(r"\s+", "", rif_raw).upper()
        else:
            m2 = re.search(r"\b([VEJPG])[-\s]?(\d{6,9})(?:[-\s]?(\d))?\b", txt_norm, flags=re.IGNORECASE)
            if m2:
                parts = [m2.group(1).upper(), m2.group(2)]
                if m2.group(3):
                    parts.append(m2.group(3))
                rif = '-'.join([p for p in parts if p])
        name = None
        addr = None
        lines = txt_norm.splitlines()
        for i, line in enumerate(lines):
            low = line.lower()
            if 'nombre' in low:
                if ':' in line:
                    name = line.split(':', 1)[1].strip()
                elif i + 1 < len(lines):
                    name = lines[i + 1].strip()
                else:
                    name = line.strip()
            if any(k in low for k in ('dirección', 'direccion', 'domicilio', 'dir')):
                if ':' in line:
                    addr = line.split(':', 1)[1].strip()
                elif i + 1 < len(lines):
                    addr = lines[i + 1].strip()
                else:
                    addr = line.strip()
        if not name:
            for line in lines:
                if any(ch.isdigit() for ch in line):
                    continue
                words = [w for w in line.split() if len(w) > 1]
                if 2 <= len(words) <= 4 and line == line.upper():
                    name = line.strip()
                    break
        if not addr:
            for line in lines:
                if any(tok in line.lower() for tok in ('calle', 'av.', 'avenida', 'sector', 'casa', 'zip', 'zona', 'parroquia')):
                    addr = line.strip()
                    break
        return {"dates": dates, "name": name or None, "address": addr or None, "rif": rif or None}

    def _html_to_text_local(html_text: str) -> str:
        if not html_text:
            return ''
        try:
            s = _html_lib.unescape(html_text)
        except Exception:
            s = html_text
        s = re.sub(r"(?i)<br\s*/?>", "\n", s)
        s = re.sub(r"(?i)</tr>", "\n", s)
        s = re.sub(r"(?i)</td>", "\n", s)
        s = re.sub(r"(?i)</p>", "\n", s)
        s = re.sub(r"<[^>]+>", "", s)
        s = s.replace('\xa0', ' ')
        s = re.sub(r"[ \t\x0b\f\r]+", " ", s)
        lines = [ln.strip() for ln in s.splitlines()]
        lines = [ln for ln in lines if ln]
        return '\n'.join(lines)

    def _parse_rif_html_local(html_text: str) -> Dict[str, Any]:
        text = _html_to_text_local(html_text)
        if not text:
            return {}
        out: Dict[str, Any] = {'raw_text': text}
        m = re.search(r"COMPROBANTE[^A-Z0-9\n\r\-]*([A-Z0-9\-]+)", text, flags=re.IGNORECASE)
        if m:
            out['numero_comprobante'] = m.group(1).strip()
        rif_match = re.search(r"\b([VEJPG])[-\s]?(\d{6,9})(?:[-\s]?(\d))?\b", text, flags=re.IGNORECASE)
        if rif_match:
            parts = [rif_match.group(1).upper(), rif_match.group(2)]
            if rif_match.group(3):
                parts.append(rif_match.group(3))
            out['rif'] = '-'.join(parts)
            for line in text.splitlines():
                if rif_match.group(2) in line:
                    line_un = line
                    try:
                        idx = line_un.find(rif_match.group(0))
                    except Exception:
                        idx = -1
                    if idx != -1:
                        name_part = line_un[idx + len(rif_match.group(0)):].strip()
                        if name_part:
                            out['nombre'] = name_part
                    break
        m_dom = re.search(r"DOMICILIO\s+FISCAL\s*(.*?)(?:ZONA\s+POSTAL|ZONA\s+POSTAL|ZONA|$)", text, flags=re.IGNORECASE | re.DOTALL)
        if m_dom:
            dom = m_dom.group(1).strip()
            dom = re.sub(r"\s+", " ", dom)
            out['domicilio'] = dom
        m_zp = re.search(r"ZONA\s+POSTAL\s*[:\s]*([0-9]{3,6})", text, flags=re.IGNORECASE)
        if m_zp:
            out['zona_postal'] = m_zp.group(1)
        ger_line = None
        for line in text.splitlines():
            if 'GERENCIA' in line.upper() or 'SEDE REGIONAL' in line.upper():
                ger_line = line.strip()
                break
        if ger_line:
            out['gerencia'] = ger_line
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if 'FIRMA AUTORIZADA' in line.upper():
                code = None
                mcode = re.search(r"([0-9]{6,}-[A-Z0-9]{1,6})", line)
                if mcode:
                    code = mcode.group(1)
                else:
                    if i > 0:
                        prev = lines[i-1]
                        m2 = re.search(r"([0-9]{6,}-[A-Z0-9]{1,6})", prev)
                        if m2:
                            code = m2.group(1)
                if code:
                    out['firma_autorizada'] = code
                break
        date_pat = r"\b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}\b"
        def find_date_after(label: str) -> Optional[str]:
            lab = label.lower()
            lo = text.lower()
            idx = lo.find(lab)
            if idx == -1:
                return None
            tail = text[idx:idx+200]
            m = re.search(date_pat, tail)
            if m:
                return m.group(0)
            m2 = re.search(date_pat, text[idx:idx+400])
            if m2:
                return m2.group(0)
            return None
        fi = find_date_after('FECHA DE INSCRIPCI') or find_date_after('FECHA DE INSCRIPCION')
        fu = find_date_after('FECHA DE &Uacute;LTIMA') or find_date_after('FECHA DE ULTIMA') or find_date_after('FECHA DE ÚLTIMA')
        fv = find_date_after('FECHA DE VENCIMIENTO')
        if fi:
            out['fecha_inscripcion'] = fi
        if fu:
            out['fecha_ultima_actualizacion'] = fu
        if fv:
            out['fecha_vencimiento'] = fv
        return out

    def _fetch_and_extract_local(url: str, max_size_local: int = max_size) -> Dict[str, Any]:
        res: Dict[str, Any] = {'url': url}
        headers = {'User-Agent': 'identification-checker/1.0'}
        with requests.get(url, headers=headers, stream=True, timeout=10) as r:
            res['status_code'] = r.status_code
            content_type = r.headers.get('content-type', '')
            res['content_type'] = content_type
            data = bytearray()
            total = 0
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    data.extend(chunk)
                    total += len(chunk)
                    if total > max_size_local:
                        res['error'] = 'download exceeds max_size'
                        res['size'] = total
                        return res
        res['size'] = total
        lc = res.get('content_type', '').lower()
        if 'pdf' in lc or url.lower().endswith('.pdf'):
            res['error'] = 'pdf content not supported'
            return res
        if 'image' in lc or any(url.lower().endswith(ext) for ext in ('.png', '.jpg', '.jpeg', '.webp')):
            try:
                img = Image.open(io.BytesIO(bytes(data)))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                try:
                    txt = pytesseract.image_to_string(img)
                    res['extracted_text'] = txt
                    if txt and txt.strip():
                        res['extracted_fields'] = _parse_fields_from_text_local(txt)
                except Exception as e:
                    res['ocr_error'] = str(e)
                return res
            except Exception as e:
                res['error'] = f'image open failed: {e}'
                return res
        if lc.startswith('text') or 'json' in lc:
            try:
                text = bytes(data).decode('utf-8', errors='replace')
                res['extracted_text'] = text
                if '<html' in text.lower() or '<body' in text.lower():
                    try:
                        parsed = _parse_rif_html_local(text)
                        if parsed:
                            res['extracted_fields'] = parsed
                    except Exception:
                        pass
                return res
            except Exception as e:
                res['error'] = f'text decode failed: {e}'
                return res
        res['note'] = 'unknown content-type, returned size only'
        return res

    # procesamiento principal
    try:
        if filename and filename.lower().endswith('.pdf'):
            return {'ok': False, 'error': 'pdf files are not supported, please upload an image', 'results': []}
        img = _image_from_bytes_local(data)
        if img is None:
            return {'ok': False, 'error': 'invalid image file', 'results': []}
        results = _detect_qr_local(img)
        if any(isinstance(r, dict) and r.get('error') for r in results):
            return {'ok': False, 'error': results[0].get('error'), 'results': []}
        def looks_like_url(s: str) -> bool:
            return isinstance(s, str) and (s.startswith('http://') or s.startswith('https://'))
        if follow_remote:
            for r in results:
                data_field = r.get('data') if isinstance(r, dict) else None
                if looks_like_url(data_field):
                    try:
                        remote = _fetch_and_extract_local(data_field)
                        r['remote'] = remote
                    except Exception as e:
                        r['remote'] = {'error': str(e)}
        return {'ok': True, 'error': None, 'results': results}
    except Exception as e:
        return {'ok': False, 'error': str(e), 'results': []}

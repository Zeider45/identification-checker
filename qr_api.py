import io
# importar el módulo sys para leer argumentos y controlar el arranque
import sys
# importar tipos para anotaciones: listas, diccionarios, Any, Optional, Set y Tuple
from typing import List, Dict, Any, Optional, Set, Tuple
import re
import html as _html_lib

# Flask: framework web ligero
from flask import Flask, request, jsonify
# OpenCV para procesamiento de imagenes y detección de QR
import cv2
# numpy para manipular arrays de imagen
import numpy as np
# requests para bajar recursos remotos
import requests
# Pillow para abrir imágenes (fallback/ocr)
from PIL import Image
# pytesseract para OCR en imágenes
import pytesseract
# math para operaciones matemáticas (si se necesitara)
import math

# nota: pyzbar se importa dinámicamente dentro de la función de fallback
# para evitar requerir la librería nativa zbar al importar el módulo.

# crear la aplicación Flask
app = Flask(__name__)


def _image_from_bytes(data: bytes) -> Optional[np.ndarray]:
    """Decodifica bytes de imagen en un array de OpenCV (BGR).

    Devuelve None si la decodificación falla.
    """
    # crea un array de bytes sin copia
    arr = np.frombuffer(data, dtype=np.uint8)
    # decodifica la imagen usando OpenCV y obtiene una imagen BGR
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # devuelve None si la imagen no pudo decodificarse
    return img


def _detect_qr(img: np.ndarray) -> List[Dict[str, Any]]:
    """Detecta y decodifica códigos QR en una imagen usando OpenCV y fallback.

    Devuelve una lista de diccionarios con keys: data (texto) y points (coordenadas).
    """
    # lista donde guardamos los resultados encontrados
    results: List[Dict[str, Any]] = []
    # si la imagen es None, retornamos lista vacía
    if img is None:
        return results

    # función auxiliar que intenta detectar con OpenCV
    def try_opencv(image: np.ndarray) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        # instanciamos el detector de QR de OpenCV
        qr = cv2.QRCodeDetector()

        # si la implementación soporta detectAndDecodeMulti, la usamos
        if hasattr(qr, "detectAndDecodeMulti"):
            try:
                # detectAndDecodeMulti devuelve ok, decoded_info, points, straight_qrcode
                ok, decoded_info, points, _ = qr.detectAndDecodeMulti(image)
            except Exception:
                # en caso de error, marcamos como no ok
                ok = False
                decoded_info = []
                points = None

            # si hubo resultados, los agregamos a la lista de salida
            if ok and decoded_info:
                for idx, text in enumerate(decoded_info):
                    # ignorar entradas vacías
                    if not text:
                        continue
                    pts = None
                    try:
                        # puntos pueden venir como array; los convertimos a lista
                        if points is not None and len(points) > idx:
                            pts = points[idx].tolist()
                    except Exception:
                        pts = None
                    out.append({"data": text, "points": pts})
                return out

        # fallback: intentar detección simple (único QR)
        try:
            text, pts = qr.detectAndDecode(image)
            if text:
                pts_list = None
                # pts puede ser None o un array; comprobamos su tamaño
                if pts is not None and getattr(pts, "size", 0):
                    pts_list = pts.tolist()
                out.append({"data": text, "points": pts_list})
        except Exception:
            # ignorar errores internos de OpenCV
            pass

        return out

    # función auxiliar que intenta detectar con pyzbar (fallback)
    def try_pyzbar(image: np.ndarray) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        try:
            # import dinámico para evitar forzar la dependencia si no está instalada
            from pyzbar import pyzbar
        except Exception:
            # si pyzbar no está disponible, retorna lista vacía
            return out

        # decodifica usando pyzbar (acepta imagen numpy)
        decs = pyzbar.decode(image)
        for d in decs:
            try:
                # intentamos construir una lista de puntos (x,y)
                pts = [(p.x, p.y) for p in d.polygon] if d.polygon else None
            except Exception:
                pts = None
            try:
                # decodificamos bytes a utf-8
                data = d.data.decode("utf-8")
            except Exception:
                # si falla, decodificamos ignorando errores
                data = d.data.decode(errors="ignore")
            out.append({"data": data, "points": pts})
        return out

    # generar variantes preprocesadas para mejorar la detección
    def generate_variants(image: np.ndarray) -> List[np.ndarray]:
        variants: List[np.ndarray] = []

        # añadimos la imagen original
        variants.append(image)
        # convertimos a escala de grises
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variants.append(gray)

        # ecualización de histograma en la imagen en escala de grises
        try:
            eq = cv2.equalizeHist(gray)
            variants.append(eq)
        except Exception:
            pass

        # umbral adaptativo para resaltar bordes y patrones
        try:
            at = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 15, 8)
            variants.append(at)
        except Exception:
            pass

        # desenfoque seguido de umbral (Otsu)
        try:
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(th)
        except Exception:
            pass

        # diferentes escalados para detectar QR pequeños
        h, w = gray.shape[:2]
        for scale in (1.5, 2.0):
            try:
                nw, nh = int(w * scale), int(h * scale)
                resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
                variants.append(resized)
            except Exception:
                pass

        # pequeñas rotaciones para cubrir QR inclinados
        angles = (-15, 15, -7, 7)
        for a in angles:
            try:
                M = cv2.getRotationMatrix2D((w // 2, h // 2), a, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE)
                variants.append(rotated)
            except Exception:
                pass

        # flips horizontales y verticales
        try:
            variants.append(cv2.flip(image, 1))
            variants.append(cv2.flip(image, 0))
        except Exception:
            pass

        # eliminar duplicados por forma/dimensión para ahorrar trabajo
        uniq: List[np.ndarray] = []
        seen: Set[Tuple[int, int, int]] = set()
        for v in variants:
            key = (v.shape[0], v.shape[1], getattr(v, "ndim", 2))
            if key not in seen:
                seen.add(key)
                uniq.append(v)

        return uniq

    # conjunto para evitar duplicar textos ya encontrados
    seen_texts: Set[str] = set()

    # generamos variantes y probamos cada una
    variants = generate_variants(img)
    # iteramos variantes: primero OpenCV, luego pyzbar
    for var in variants:
        # intentar con OpenCV
        try:
            op_results = try_opencv(var)
        except Exception:
            op_results = []

        # añadimos resultados de OpenCV si no están repetidos
        for r in op_results:
            txt = r.get("data")
            if txt and txt not in seen_texts:
                seen_texts.add(txt)
                results.append(r)

        # intentar con pyzbar como complemento
        try:
            p_results = try_pyzbar(var)
        except Exception:
            p_results = []

        for r in p_results:
            txt = r.get("data")
            if txt and txt not in seen_texts:
                seen_texts.add(txt)
                results.append(r)

        # si ya encontramos al menos un QR, detenemos búsqueda para ahorrar tiempo
        if results:
            break

    return results


def _parse_fields_from_text(text: str) -> Dict[str, Any]:
    """Intenta extraer campos útiles del texto OCR: fechas, nombre, dirección y número de RIF.

    Devuelve un dict con keys: 'dates' (lista), 'name' (str|None), 'address' (str|None), 'rif' (str|None).
    Usa heurísticas simples: busca etiquetas comunes ('nombre', 'dirección', 'rif') y patrones regex.
    """
    if not text:
        return {"dates": [], "name": None, "address": None, "rif": None}

    txt = text.replace('\r', '\n')
    # normalizar espacios
    txt_norm = '\n'.join(line.strip() for line in txt.splitlines() if line.strip())
    lower = txt_norm.lower()

    # buscar fechas comunes (DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, DD.MM.YYYY)
    import re
    dates = []
    date_patterns = [r"\b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}\b", r"\b\d{4}-\d{2}-\d{2}\b"]
    for pat in date_patterns:
        for m in re.findall(pat, txt_norm):
            if m not in dates:
                dates.append(m)

    # buscar RIF: etiqueta explícita o patrón (V/J/E/P/G etc.)
    rif = None
    # intento con etiqueta
    m = re.search(r"rif[:\s]*([VEJPG\-\s]?\d{6,9}-?\d?)", lower, flags=re.IGNORECASE)
    if m:
        rif_raw = m.group(1)
        rif = re.sub(r"\s+", "", rif_raw).upper()
    else:
        # patrón suelto: letra opcional + números
        m2 = re.search(r"\b([VEJPG])[-\s]?(\d{6,9})(?:[-\s]?(\d))?\b", txt_norm, flags=re.IGNORECASE)
        if m2:
            parts = [m2.group(1).upper(), m2.group(2)]
            if m2.group(3):
                parts.append(m2.group(3))
            rif = '-'.join([p for p in parts if p])

    # buscar nombre por etiqueta
    name = None
    addr = None
    lines = txt_norm.splitlines()
    for i, line in enumerate(lines):
        low = line.lower()
        if 'nombre' in low:
            # tomar la parte después de ':' si existe, sino la línea siguiente
            if ':' in line:
                name = line.split(':', 1)[1].strip()
            elif i + 1 < len(lines):
                name = lines[i + 1].strip()
            else:
                # fallback: todo el line
                name = line.strip()
        if any(k in low for k in ('dirección', 'direccion', 'domicilio', 'dir')):
            if ':' in line:
                addr = line.split(':', 1)[1].strip()
            elif i + 1 < len(lines):
                addr = lines[i + 1].strip()
            else:
                addr = line.strip()

    # heurística adicional para nombre (línea en mayúsculas sin dígitos con 2-4 palabras)
    if not name:
        for line in lines:
            if any(ch.isdigit() for ch in line):
                continue
            words = [w for w in line.split() if len(w) > 1]
            if 2 <= len(words) <= 4 and line == line.upper():
                name = line.strip()
                break

    # heurística adicional para dirección (línea que contiene 'Av.' 'Calle' 'Sector' 'Casa' o números y letras)
    if not addr:
        for line in lines:
            if any(tok in line.lower() for tok in ('calle', 'av.', 'avenida', 'sector', 'casa', 'zip', 'zona', 'parroquia')):
                addr = line.strip()
                break

    return {"dates": dates, "name": name or None, "address": addr or None, "rif": rif or None}


def _html_to_text(html_text: str) -> str:
    """Convierte HTML simple a texto plano conservando saltos de línea útiles.

    Reemplaza <br>, </tr>, </td>, </p> por saltos de línea antes de eliminar tags.
    Luego desescapea entidades HTML.
    """
    if not html_text:
        return ""

    # unescape primero para manejar &nbsp; y caracteres especiales
    try:
        s = _html_lib.unescape(html_text)
    except Exception:
        s = html_text

    # normalizar saltos de línea en etiquetas comunes
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"(?i)</tr>", "\n", s)
    s = re.sub(r"(?i)</td>", "\n", s)
    s = re.sub(r"(?i)</p>", "\n", s)

    # eliminar cualquier tag restante
    s = re.sub(r"<[^>]+>", "", s)

    # convertir múltiples espacios y &nbsp; a espacios simples
    s = s.replace('\xa0', ' ')
    s = re.sub(r"[ \t\x0b\f\r]+", " ", s)

    # normalizar líneas: limpiar espacios en cada línea y eliminar vacías
    lines = [ln.strip() for ln in s.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def _parse_rif_html(html_text: str) -> Dict[str, Any]:
    """Extrae campos estructurados del HTML retornado por el SENIAT para RIF.

    Campos extraídos (cuando estén disponibles):
      - numero_comprobante
      - rif
      - nombre
      - domicilio
      - zona_postal
      - fecha_inscripcion
      - fecha_ultima_actualizacion
      - fecha_vencimiento
      - gerencia
      - firma_autorizada (código si aparece)
      - raw_text

    La función es tolerante y usa heurísticas sobre el texto plano generado desde el HTML.
    """
    text = _html_to_text(html_text)
    if not text:
        return {}

    out: Dict[str, Any] = {"raw_text": text}

    # buscar número de comprobante
    m = re.search(r"COMPROBANTE[^A-Z0-9\n\r\-]*([A-Z0-9\-]+)", text, flags=re.IGNORECASE)
    if m:
        out["numero_comprobante"] = m.group(1).strip()

    # buscar RIF (primera ocurrencia de formato: letra + números)
    rif_match = re.search(r"\b([VEJPG])[-\s]?(\d{6,9})(?:[-\s]?(\d))?\b", text, flags=re.IGNORECASE)
    if rif_match:
        parts = [rif_match.group(1).upper(), rif_match.group(2)]
        if rif_match.group(3):
            parts.append(rif_match.group(3))
        rif_val = "-".join(parts)
        out["rif"] = rif_val

        # intentar tomar el nombre que sigue en la misma línea
        for line in text.splitlines():
            if rif_match.group(2) in line:
                # encontrar la porción después del rif dentro de la línea
                # buscamos la posición del patrón en la línea raw
                line_un = line
                try:
                    # ubicar la ocurrencia del número dentro de la línea
                    idx = line_un.find(rif_match.group(0))
                except Exception:
                    idx = -1
                if idx != -1:
                    name_part = line_un[idx + len(rif_match.group(0)):].strip()
                    if name_part:
                        out["nombre"] = name_part
                break

    # domicilio fiscal
    m_dom = re.search(r"DOMICILIO\s+FISCAL\s*(.*?)(?:ZONA\s+POSTAL|ZONA\s+POSTAL|ZONA|$)", text, flags=re.IGNORECASE | re.DOTALL)
    if m_dom:
        dom = m_dom.group(1).strip()
        # limpiar espacios excesivos
        dom = re.sub(r"\s+", " ", dom)
        out["domicilio"] = dom

    # zona postal
    m_zp = re.search(r"ZONA\s+POSTAL\s*[:\s]*([0-9]{3,6})", text, flags=re.IGNORECASE)
    if m_zp:
        out["zona_postal"] = m_zp.group(1)

    # gerencia / sede
    ger_line = None
    for line in text.splitlines():
        if 'GERENCIA' in line.upper() or 'SEDE REGIONAL' in line.upper():
            ger_line = line.strip()
            break
    if ger_line:
        out["gerencia"] = ger_line

    # firma autorizada: buscar código cercano a la etiqueta
    # buscar la línea que contiene FIRMA AUTORIZADA y tomar la línea anterior si tiene patrón con guion
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if 'FIRMA AUTORIZADA' in line.upper():
            # buscar en la misma línea algún token con dígitos y guion
            code = None
            mcode = re.search(r"([0-9]{6,}-[A-Z0-9]{1,6})", line)
            if mcode:
                code = mcode.group(1)
            else:
                # revisar línea previa
                if i > 0:
                    prev = lines[i-1]
                    m2 = re.search(r"([0-9]{6,}-[A-Z0-9]{1,6})", prev)
                    if m2:
                        code = m2.group(1)
            if code:
                out["firma_autorizada"] = code
            break

    # buscar fechas específicas (inscripción, última actualización, vencimiento)
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
        # fallback: buscar en todo el texto cercano
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


# Nota: soporte de PDF eliminado. Solo se aceptan imágenes.


def _fetch_and_extract(url: str, max_size: int = 20 * 1024 * 1024) -> Dict[str, Any]:
    """Descarga la URL y trata de extraer contenido útil.

    Devuelve un diccionario con keys: url, status_code, content_type, size y extracted_text opcional.
    """
    res: Dict[str, Any] = {"url": url}
    # encabezado sencillo para peticiones
    headers = {"User-Agent": "identification-checker/1.0"}
    # hacemos la petición en streaming para limitar tamaño
    with requests.get(url, headers=headers, stream=True, timeout=10) as r:
        res["status_code"] = r.status_code
        content_type = r.headers.get("content-type", "")
        res["content_type"] = content_type
        # leemos hasta max_size bytes
        data = bytearray()
        total = 0
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                data.extend(chunk)
                total += len(chunk)
                if total > max_size:
                    # si excede, devolvemos error y tamaño parcial
                    res["error"] = "download exceeds max_size"
                    res["size"] = total
                    return res

    res["size"] = total

    # decidir cómo extraer según el content-type o extensión
    lc = content_type.lower()
    # PDF ya no está soportado: devolvemos un mensaje claro
    if "pdf" in lc or url.lower().endswith(".pdf"):
        res["error"] = "pdf content not supported"
        return res

    # si es imagen, intentar OCR con pytesseract
    if "image" in lc or any(url.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")):
        try:
            img = Image.open(io.BytesIO(bytes(data)))
            # convertir a RGB si es necesario
            if img.mode != "RGB":
                img = img.convert("RGB")
            try:
                txt = pytesseract.image_to_string(img)
                res["extracted_text"] = txt
                # parsear campos estructurados si hay texto
                if txt and txt.strip():
                    res["extracted_fields"] = _parse_fields_from_text(txt)
            except Exception as e:
                # si falla OCR, registramos el error
                res["ocr_error"] = str(e)
            return res
        except Exception as e:
            res["error"] = f"image open failed: {e}"
            return res

    # si es texto o JSON, decodificamos como UTF-8
    if lc.startswith("text") or "json" in lc:
        try:
            text = bytes(data).decode("utf-8", errors="replace")
            res["extracted_text"] = text
            # si parece HTML, intentar parsearlo como tal para extraer campos del RIF
            if "<html" in text.lower() or "<body" in text.lower():
                try:
                    parsed = _parse_rif_html(text)
                    # incluir parsed sólo si hay algún campo útil
                    if parsed:
                        res["extracted_fields"] = parsed
                except Exception:
                    # no fallar en caso de parsing
                    pass

            return res
        except Exception as e:
            res["error"] = f"text decode failed: {e}"
            return res

    # fallback: no sabemos el tipo, devolvemos metadatos
    res["note"] = "unknown content-type, returned size only"
    return res


@app.route("/scan", methods=["POST"])
def scan():
    """Acepta multipart/form-data con un archivo (pdf o imagen), busca QR y retorna JSON."""
    # si no se proporcionó campo 'file', devolvemos error 400
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "no file provided", "results": []}), 400

    # obtenemos el archivo subido
    f = request.files["file"]
    # normalizamos el nombre a minúsculas
    filename = (f.filename or "").lower()
    # leemos todo el contenido en memoria (para uso simple)
    data = f.read()

    results: List[Dict[str, Any]] = []

    try:
        # No se aceptan PDFs: sólo imágenes
        if filename.endswith(".pdf") or f.mimetype == "application/pdf":
            return jsonify({"ok": False, "error": "pdf files are not supported, please upload an image", "results": []}), 400

        # si es una imagen, intentamos decodificar y detectar QR
        img = _image_from_bytes(data)
        if img is None:
            return jsonify({"ok": False, "error": "invalid image file", "results": []}), 400
        results = _detect_qr(img)

        # si cualquier resultado contiene un error, respondemos con 400
        if any(isinstance(r, dict) and r.get("error") for r in results):
            return jsonify({"ok": False, "error": results[0].get("error"), "results": []}), 400

        # función auxiliar para detectar si un texto es una URL
        def looks_like_url(s: str) -> bool:
            return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))

        # para cada resultado que parezca una URL, seguimos la URL y extraemos su contenido
        for r in results:
            data_field = r.get("data") if isinstance(r, dict) else None
            if looks_like_url(data_field):
                try:
                    remote = _fetch_and_extract(data_field)
                    r["remote"] = remote
                except Exception as e:
                    r["remote"] = {"error": str(e)}

        # respondemos con los resultados
        return jsonify({"ok": True, "error": None, "results": results}), 200
    except Exception as e:
        # error inesperado
        return jsonify({"ok": False, "error": str(e), "results": []}), 500


if __name__ == "__main__":
    # Ejecutar servidor de desarrollo
    port = 5000
    # permitir pasar puerto como primer argumento
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except Exception:
            pass
    # iniciar Flask
    app.run(host="0.0.0.0", port=port, debug=True)

import io
# importar el módulo sys para leer argumentos y controlar el arranque
import sys
# importar tipos para anotaciones: listas, diccionarios, Any, Optional, Set y Tuple
from typing import List, Dict, Any, Optional, Set, Tuple

# Flask: framework web ligero
from flask import Flask, request, jsonify
# OpenCV para procesamiento de imagenes y detección de QR
import cv2
# numpy para manipular arrays de imagen
import numpy as np
# PyMuPDF (fitz) para trabajar con PDFs
import fitz  # PyMuPDF
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


def _scan_pdf(file_bytes: bytes) -> List[Dict[str, Any]]:
    """Renderiza cada página del PDF y busca códigos QR.

    Devuelve resultados con número de página si se encuentran.
    """
    # lista donde acumulamos resultados
    results: List[Dict[str, Any]] = []
    try:
        # abrimos el PDF desde bytes
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        # si falla la apertura, devolvemos un error en la lista
        return [{"error": f"failed to open pdf: {e}"}]

    # recorremos cada página y la renderizamos a imagen
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=200)
        png_bytes = pix.tobytes("png")
        img = _image_from_bytes(png_bytes)
        page_results = _detect_qr(img)
        for r in page_results:
            # añadimos el número de página si se encontró QR
            r["page"] = i + 1
        results.extend(page_results)

    # Intentamos también extraer texto del PDF (por si el archivo contiene texto legible)
    try:
        text_pages = []
        for i in range(len(doc)):
            page = doc.load_page(i)
            text_pages.append(page.get_text("text"))
        if any(p.strip() for p in text_pages):
            results.append({"pdf_text": "\n\n".join(text_pages)})
    except Exception:
        # ignoramos errores durante la extracción de texto
        pass

    return results


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
    # si es PDF, intentar extraer texto con PyMuPDF
    if "pdf" in lc or url.lower().endswith(".pdf"):
        try:
            try:
                doc = fitz.open(stream=bytes(data), filetype="pdf")
            except Exception:
                doc = fitz.open(stream=bytes(data))
            texts = []
            for i in range(len(doc)):
                page = doc.load_page(i)
                texts.append(page.get_text("text"))
            res["extracted_text"] = "\n\n".join(texts)
            return res
        except Exception as e:
            res["error"] = f"pdf extraction failed: {e}"
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
        # si parece un PDF, lo procesamos como tal
        if filename.endswith(".pdf") or f.mimetype == "application/pdf":
            results = _scan_pdf(data)
        else:
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

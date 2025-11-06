# Identification OCR API

Pequeña API Flask que recibe una imagen (campo `image` en multipart/form-data), ejecuta OCR para detectar si la identificación contiene las palabras clave usadas originalmente, y si es válida devuelve un recorte (usando el mismo rectángulo definido por la variable `square` en el código).

Requisitos

- Python 3.8+
- Tesseract OCR instalado en Windows (la ruta está configurada en `video.py`).

Instalación rápida

```powershell
python -m pip install -r requirements.txt
```

Ejecutar localmente

```powershell
python video.py
```

Endpoint

- POST /validate
  - multipart/form-data, campo `image` = archivo de imagen
  - Response JSON:
    - valid: true/false
    - text: texto extraído por OCR
    - crop: data URI (base64 PNG) sólo presente si valid == true

Ejemplo con curl (PowerShell)

```powershell
curl -F "image=@C:\path\to\id.jpg" http://localhost:5000/validate
```

Notas

- En producción usa un servidor WSGI (gunicorn/uWSGI) o ASGI según prefieras.
- Ajusta `pytesseract.pytesseract.tesseract_cmd` en `video.py` si la ruta de Tesseract es distinta.

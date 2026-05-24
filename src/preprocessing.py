import fitz  # PyMuPDF
from PIL import Image


# ---------------------------------------------------------------
# CHANGES FROM ORIGINAL:
#   - Removed binarization (OTSU threshold) and denoising
#     → Those steps help Tesseract OCR but HURT vision LLMs
#     → LLMs read ink weight, shading, context — binarizing kills that
#   - Removed OpenCV dependency entirely (not needed anymore)
#   - DPI set to 150: sharp enough for LLM, keeps image size small
#     → 300 DPI doubles image size and cost with no quality gain for LLMs
#     → 72 DPI is too blurry for dense handwriting
# ---------------------------------------------------------------

def pdf_to_images(pdf_path: str, dpi: int = 150):
    doc = fitz.open(pdf_path)
    images = []

    for page in doc:
        # Render at specified DPI
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    doc.close()
    return images

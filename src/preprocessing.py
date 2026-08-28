import fitz
from PIL import Image


def pdf_to_images(pdf_path: str, dpi: int = 150):
    doc = fitz.open(pdf_path)
    images = []

    for page in doc:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    doc.close()
    return images

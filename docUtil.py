import fitz
from pdf2image import convert_from_path
import os

def extract_text_and_images(pdf_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    images = []
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        images.append(convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1)[0])
    return images

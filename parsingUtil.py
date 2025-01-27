import os
import pytesseract
from PIL import Image
from docUtil import extract_text_and_images

def perform_ocr_on_pdf(pdf_path, output_folder="extracted_images"):
    images = extract_text_and_images(pdf_path, output_folder)
    all_text = ""
    for idx, image in enumerate(images, start=1):
        text = pytesseract.image_to_string(image)
        all_text += f"\n--- Page {idx} ---\n{text}\n"
    return all_text

def parse_extracted_text_with_regex(extracted_text):
    import re
    date_pattern = re.compile(r'Date Issued:\s*(\d{1,2}/\d{1,2}/\d{4})')
    amount_pattern = re.compile(r'Total\s+(\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)')

    dates = date_pattern.findall(extracted_text)
    amounts = amount_pattern.findall(extracted_text)

    records = [{"Date": date, "Amount": amount} for date, amount in zip(dates, amounts)]
    return records

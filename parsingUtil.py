
import os
import re
from itertools import zip_longest

import pytesseract
from PIL import Image
from docUtil import extract_text_and_images  # Assuming 'docUtil.py' contains the function

def perform_ocr_on_pdf(pdf_path, output_folder="extracted_images"):
    """
    Extract images from a PDF and perform OCR to extract text.

    Args:
        pdf_path (str): Path to the PDF file.
        output_folder (str): Folder to save extracted images.

    Returns:
        str: The extracted text from the PDF.
    """
    # Extract images from PDF
    print("Extracting images from PDF...")
    images = extract_text_and_images(pdf_path, output_folder)

    if not images:
        print("No images were extracted from the PDF.")
        return ""

    print(f"Extracted {len(images)} image(s) from the PDF.")

    all_text = ""
    for idx, image in enumerate(images, start=1):
        print(f"Performing OCR on page {idx}...")
        try:
            # Perform OCR using pytesseract
            text = pytesseract.image_to_string(image)
            all_text += f"\n--- Page {idx} ---\n{text}\n"
        except Exception as e:
            print(f"Error performing OCR on page {idx}: {e}")

    return all_text


def save_text_to_file(text, original_pdf_path):
    """
    Save the extracted text to a .txt file.

    Args:
        text (str): The text to save.
        original_pdf_path (str): Path to the original PDF to derive the text file name.
    """
    base_name = os.path.splitext(os.path.basename(original_pdf_path))[0]
    text_file_path = os.path.join(os.path.dirname(original_pdf_path), f"{base_name}_extracted.txt")
    try:
        with open(text_file_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Extracted text saved to: {text_file_path}")
    except Exception as e:
        print(f"Failed to save text to file: {e}")


def parse_extracted_text_with_regex(extracted_text):
    """
    Parse the extracted text to find specific data using regular expressions.

    Args:
        extracted_text (str): The OCR-extracted text.

    Returns:
        list of dict: A list of dictionaries containing the extracted data.
    """
    records = []

    # Define regex patterns
    date_pattern = re.compile(r'Date Issued:\s*(\d{1,2}/\d{1,2}/\d{4})', re.IGNORECASE)
    po_pattern = re.compile(r'Purchase Order\s*-\s*(\S+)', re.IGNORECASE)
    company_pattern = re.compile(r'Bill To:\s*\n\s*(\S.*)', re.IGNORECASE)
    location_pattern = re.compile(r'Ship To/ Provide Service At:\s*\n\s*(\S.*)', re.IGNORECASE)
    amount_pattern = re.compile(r'Total\s+(\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', re.IGNORECASE)

    description_pattern = re.compile(
        r'Order Name\s*-\s*([\s\S]*?)(?=-\s*Troon)',
        re.IGNORECASE
    )

    # Extract main fields
    dates = date_pattern.findall(extracted_text)
    pos = po_pattern.findall(extracted_text)
    companies_raw = company_pattern.findall(extracted_text)
    locations = location_pattern.findall(extracted_text)
    amounts = amount_pattern.findall(extracted_text)
    descriptions = description_pattern.findall(extracted_text)

    # Debugging: Print the extracted matches
    print("\n--- Debugging Extracted Fields ---")
    print(f"Dates Found: {dates}")
    print(f"Purchase Orders Found: {pos}")
    print(f"Companies Found: {companies_raw}")
    print(f"Locations Found: {locations}")
    print(f"Amounts Found: {amounts}")
    print(f"Descriptions Found: {descriptions}")
    print("--- End of Extracted Fields ---\n")

    # Determine the number of records based on the maximum length of all lists using zip_longest
    for date, po, company_raw, location, amount, description in zip_longest(
        dates, pos, companies_raw, locations, amounts, descriptions,
        fillvalue="N/A"
    ):
        # Determine Company based on presence of 'Gateway' in 'Bill To' section
        if company_raw != "N/A":
            if 'gateway' in company_raw.lower():
                company = 'Gateway'
            else:
                company = 'Lefrak'
        else:
            company = 'N/A'

        # Create a record dictionary with all fields
        record = {
            "Date": date if date != "N/A" else "N/A",
            "Purchase Order": po if po != "N/A" else "N/A",
            "Company": company,
            "Location": location if location != "N/A" else "N/A",
            "Amount": amount if amount != "N/A" else "N/A",
            "Description": description.strip() if description != "N/A" else "N/A",
        }
        records.append(record)

    # Debugging: Print the number of records created
    print(f"Number of Records to Create: {len(records)}\n")

    return records

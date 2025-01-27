###
# THIS IS A MODULARIZED VERSION OF THE MAIN FUNCTION AS A WHOLE
# FOR US TO MAKE THIS A REUSABLE FUNCTION ON TOP OF AN EMAIL
# WE HAD TO MODULARIZE EVERYTHING FROM MAIN
###

import tkinter as tk
from tkinter import messagebox
from parsingUtil import perform_ocr_on_pdf, save_text_to_file, parse_extracted_text_with_regex
from excelUtil import save_to_excel
from excelUtil import append_to_existing_excel


def process_pdf(pdf_path):
    if not pdf_path or not pdf_path.lower().endswith(".pdf"):
        print(f"Invalid PDF path: {pdf_path}")
        return

    print(f"Processing PDF: {pdf_path}")

    # Perform OCR on the PDF
    extracted_text = perform_ocr_on_pdf(pdf_path)

    if extracted_text:
        print("\n=== Extracted Text ===")
        print(extracted_text)

        # Parse the extracted text using regex-based parser
        parsed_records = parse_extracted_text_with_regex(extracted_text)

        # Display the parsed data
        print("\n=== Parsed Data ===")
        if parsed_records:
            for idx, record in enumerate(parsed_records, start=1):
                print(f"\n--- Record {idx} ---")
                print(f"Date: {record.get('Date', 'N/A')}")
                print(f"Purchase Order: {record.get('Purchase Order', 'N/A')}")
                print(f"Company: {record.get('Company', 'N/A')}")
                print(f"Location: {record.get('Location', 'N/A')}")
                print(f"Amount: {record.get('Amount', 'N/A')}")
                print(f"Description: {record.get('Description', 'N/A')}")
        else:
            print("No parsed records found.")

        # Initialize a hidden Tkinter root for dialogs
        root = tk.Tk()
        root.withdraw()

        root.destroy()
    else:
        print("No text was extracted from the PDF.")


def process_pdf_file(pdf_path, excel_file):
    print(f"\n=== Processing PDF: {pdf_path} ===")
    extracted_text = perform_ocr_on_pdf(pdf_path)

    if extracted_text:
        print("\n=== Extracted Text ===")
        print(extracted_text)

        parsed_records = parse_extracted_text_with_regex(extracted_text)

        print("\n=== Parsed Data ===")
        if parsed_records:
            for idx, record in enumerate(parsed_records, start=1):
                print(f"\n--- Record {idx} ---")
                print(f"Date: {record.get('Date', 'N/A')}")
                print(f"Purchase Order: {record.get('Purchase Order', 'N/A')}")
                print(f"Company: {record.get('Company', 'N/A')}")
                print(f"Location: {record.get('Location', 'N/A')}")
                print(f"Amount: {record.get('Amount', 'N/A')}")
                print(f"Description: {record.get('Description', 'N/A')}")
        else:
            print("No parsed records found.")

        # Automatically append to Excel without confirmation
        if parsed_records and excel_file:
            append_to_existing_excel(parsed_records, excel_file)
    else:
        print(f"No text was extracted from {pdf_path}.")

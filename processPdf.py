from parsingUtil import perform_ocr_on_pdf, parse_extracted_text_with_regex
from excelUtil import append_to_existing_excel

def process_pdf_file(pdf_path, excel_file):
    extracted_text = perform_ocr_on_pdf(pdf_path)
    parsed_records = parse_extracted_text_with_regex(extracted_text)
    if parsed_records and excel_file:
        append_to_existing_excel(parsed_records, excel_file)

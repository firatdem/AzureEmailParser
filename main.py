import os
from dotenv import load_dotenv
from tkinter import filedialog
from processPdf import process_pdf_file
from emailListener import continuous_listener
from excelUtil import append_to_existing_excel

# Load environment variables
load_dotenv()

def main():
    print("Select an option:")
    print("1. Parse a local file")
    print("2. Listen on an email")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        pdf_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if not pdf_path:
            print("No file selected. Exiting.")
            return

        print("Select an Excel file to append the data to (or cancel to create a new one).")
        excel_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])

        if not excel_path:
            print("No existing Excel file selected, a new one will be created.")
            process_pdf_file(pdf_path, None)
        else:
            process_pdf_file(pdf_path, excel_path)

    elif choice == "2":
        print("Starting email listener...")
        continuous_listener()

    else:
        print("Invalid choice. Please select either 1 or 2.")

if __name__ == "__main__":
    main()

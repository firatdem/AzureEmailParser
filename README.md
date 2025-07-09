
# PDF Invoice Parser & OCR Excel Exporter
```bash
[Email Inbox] → [Download PDFs] → [OCR + Regex Parsing] → [Excel Export]

                                        ↓
                               [Raw Text File Saved]
```

A Python utility built for internal use to automate the extraction of invoice data from PDFs using OCR and regex. Designed to work with a standardized invoice format, but customizable with your own regex rules for broader compatibility.

## Use Cases

- Auto-extract invoices from vendor emails
- Convert scanned PDFs into Excel records
- Speed up AP/AR workflows with minimal human input
- Internal back-office automation with GUI + email integration

## Features

- Parses PDFs using OCR (via Tesseract) and custom regex
- Outputs structured invoice data to a formatted Excel spreadsheet
- Supports batch processing from emails (via Microsoft Graph API)
- Flexible architecture allows you to adapt regex parsing logic for your own document types
- Saves raw OCR text and optionally marks up images for visual debugging
- Built-in GUI file selection dialogs

## How It Works (in Brief)

1. Downloads PDF invoices (from email or file picker)
2. Applies OCR to extract raw text from each page
3. Parses fields using regex rules
4. Saves structured data to Excel and raw text for backup

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set up `.env` file:

```env
CLIENT_ID=""
TENANT_ID=""           # Optional; defaults to 'common'
SCOPE="Mail.ReadWrite" # Can be adjusted based on your app permissions
DOWNLOAD_FOLDER="./Downloads"
POLL_INTERVAL=300      # Optional; default 300 seconds (5 min)
```

Required for Microsoft Graph API access.

## How to Use

Run the main script:

```bash
python main.py
```

You will be prompted with two options:

### Option 1: Parse Local PDF Files

- Select a PDF file
- Then choose an existing Excel file to append to, or cancel to create a new one

### Option 2: Listen to Incoming Emails

- Requires you to sign into your Microsoft 365 account using device code login
- Connects to your registered Azure App and polls for unread emails with PDF attachments
- Extracted data is appended to the Excel file you select or creates a new one

## Output

- Extracted data is saved as:
  - `*_extracted.xlsx` — Clean structured data
  - `*_extracted.txt` — Raw OCR text
- For each invoice, the following fields are extracted (if available):
  - `Date`
  - `Company`
  - `Location`
  - `Purchase Order`
  - `Description`
  - `Amount`

## Customizing for Your PDFs

The regex logic is located in:

```
parsingUtil.py > parse_extracted_text_with_regex()
```

You can modify these expressions to fit different invoice layouts. If your documents differ significantly, consider using the `parse_extracted_text()` function for more manual parsing.

## Email Listener (Graph API)

Email functionality is provided via Microsoft Graph API and is available in:

- `emailListener.py`: For continuous or batch fetching of attachments
- `emailUtil.py`: (optional IMAP listener — legacy and basic)

To use it, register your application in [Azure App Registrations](https://portal.azure.com/#blade/Microsoft_AAD_RegisteredApps) and enable required scopes.

## Notes

- If your invoice PDF is image-based, OCR will be used to extract text.
- Program assumes a consistent invoice layout; validation messages will appear in the console if key fields are missing.
- Email polling is basic but extendable. Contributions welcome.

## Requirements

See `requirements.txt`, or install manually:

```bash
pandas
openpyxl
python-dotenv
pytesseract
pillow
numpy
pymupdf
pdf2image
msal
requests
```

## Acknowledgements

Built to streamline invoice processing at work. Modular and extensible for others to adapt and improve upon.

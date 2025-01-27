import msal
import requests
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv  # Import the dotenv library
from processPdf import process_pdf
from parsingUtil import perform_ocr_on_pdf, save_text_to_file, parse_extracted_text_with_regex
from excelUtil import append_to_existing_excel  # Changed from save_to_excel to append_to_existing_excel

# Load environment variables from .env file
load_dotenv()

# Read sensitive data from environment variables
CLIENT_ID = os.getenv("CLIENT_ID")
TENANT_ID = os.getenv("TENANT_ID", "common")  # Use "common" as default if not set
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPE = os.getenv("SCOPE", "Mail.ReadWrite").split(",")
DOWNLOAD_FOLDER = os.getenv("DOWNLOAD_FOLDER", "./Downloads")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 300))  # Interval in seconds, default to 300

# Initialize a global MSAL PublicClientApplication with a token cache
app = msal.PublicClientApplication(client_id=CLIENT_ID, authority=AUTHORITY)


def get_access_token():
    """
    Try to acquire token silently from cache. If that fails, initiate device flow.
    """
    accounts = app.get_accounts()
    result = None

    # Attempt to acquire token silently if accounts exist in cache
    if accounts:
        result = app.acquire_token_silent(SCOPE, account=accounts[0])

    # If silent acquisition failed, initiate device flow
    if not result:
        flow = app.initiate_device_flow(scopes=SCOPE)
        if "user_code" not in flow:
            raise Exception("Failed to create device flow. Check your client ID and scopes.")
        print(flow["message"])
        result = app.acquire_token_by_device_flow(flow)

    if "access_token" in result:
        return result["access_token"]
    else:
        raise Exception(f"Could not obtain access token: {result.get('error_description')}")


def fetch_emails_with_attachments(access_token, additional_filter=""):
    """Use Microsoft Graph API to fetch messages that have attachments,
    optionally applying an additional filter."""
    headers = {"Authorization": f"Bearer {access_token}"}
    base_filter = "hasAttachments eq true"
    filter_query = f"{base_filter} {additional_filter}".strip()
    url = f"https://graph.microsoft.com/v1.0/me/messages?$filter={filter_query}"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def download_attachment(access_token, message_id, attachment_id, download_folder):
    """Download attachment using Microsoft Graph API."""
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"https://graph.microsoft.com/v1.0/me/messages/{message_id}/attachments/{attachment_id}/$value"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    disposition = response.headers.get("Content-Disposition", "")
    filename = "attachment.pdf"
    if "filename=" in disposition:
        filename = disposition.split("filename=")[-1].strip('"')
    filepath = os.path.join(download_folder, filename)
    with open(filepath, "wb") as f:
        f.write(response.content)
    return filepath


def fetch_pdfs_from_past_day_list():
    """Fetch all PDFs from emails received in the past day and return a list of file paths."""
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    token = get_access_token()
    past_day = datetime.utcnow() - timedelta(days=1)
    past_day_iso = past_day.isoformat() + "Z"
    additional_filter = f"and receivedDateTime ge {past_day_iso}"
    messages_data = fetch_emails_with_attachments(token, additional_filter=additional_filter)

    pdf_paths = []

    if "value" in messages_data and messages_data["value"]:
        for message in messages_data["value"]:
            message_id = message["id"]
            if not message.get("hasAttachments"):
                continue

            headers = {"Authorization": f"Bearer {token}"}
            attachments_url = f"https://graph.microsoft.com/v1.0/me/messages/{message_id}/attachments"
            attachments_response = requests.get(attachments_url, headers=headers)
            attachments_response.raise_for_status()
            attachments = attachments_response.json().get("value", [])

            for attachment in attachments:
                if attachment.get("name", "").lower().endswith(".pdf"):
                    attachment_id = attachment["id"]
                    print(f"Downloading attachment {attachment['name']} from message {message_id}")
                    pdf_path = download_attachment(token, message_id, attachment_id, DOWNLOAD_FOLDER)
                    pdf_paths.append(pdf_path)
    else:
        print("No matching messages found.")

    return pdf_paths


def process_messages(messages_data, token, excel_path):
    """Process messages to find and handle PDF attachments."""
    if "value" not in messages_data or not messages_data["value"]:
        print("No matching messages found.")
        return

    for message in messages_data["value"]:
        message_id = message["id"]
        if not message.get("hasAttachments"):
            continue

        headers = {"Authorization": f"Bearer {token}"}
        attachments_url = f"https://graph.microsoft.com/v1.0/me/messages/{message_id}/attachments"
        attachments_response = requests.get(attachments_url, headers=headers)
        attachments_response.raise_for_status()
        attachments = attachments_response.json().get("value", [])

        for attachment in attachments:
            if attachment.get("name", "").lower().endswith(".pdf"):
                attachment_id = attachment["id"]
                print(f"Downloading attachment {attachment['name']} from message {message_id}")
                pdf_path = download_attachment(token, message_id, attachment_id, DOWNLOAD_FOLDER)
                try:
                    processed_data = process_pdf(pdf_path)  # Assuming process_pdf returns a dict
                    append_to_existing_excel(processed_data, excel_path)
                    print(f"Appended data from {pdf_path} to Excel successfully.")
                except Exception as e:
                    print(f"Error processing {pdf_path}: {e}")


def continuous_listener(excel_path):
    """Continuously listen for new incoming emails and append data to Excel."""
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

    while True:
        print("Checking for new PDF attachments in email with Microsoft Graph API...")
        try:
            token = get_access_token()

            # Apply additional filter for unread emails with attachments
            additional_filter = "and isRead eq false"
            messages_data = fetch_emails_with_attachments(token, additional_filter=additional_filter)
            process_messages(messages_data, token, excel_path)

            print(f"Sleeping for {POLL_INTERVAL} seconds before next check...\n")
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Retrying in {POLL_INTERVAL} seconds...\n")
            time.sleep(POLL_INTERVAL)


def fetch_pdfs_from_past_day(excel_path):
    """Fetch all PDFs from emails received in the past day and append data to Excel."""
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    try:
        token = get_access_token()

        past_day = datetime.utcnow() - timedelta(days=1)
        past_day_iso = past_day.isoformat() + "Z"
        additional_filter = f"and receivedDateTime ge {past_day_iso}"

        messages_data = fetch_emails_with_attachments(token, additional_filter=additional_filter)
        process_messages(messages_data, token, excel_path)
    except Exception as e:
        print(f"An error occurred while fetching PDFs: {e}")


if __name__ == "__main__":
    print("Choose an option:")
    print("1: Continuously listen for new incoming emails.")
    print("2: Fetch all PDFs from the past day.")
    print("3: (Optional) Provide Excel file path for appending data.")
    choice = input("Enter 1 or 2: ").strip()

    # Depending on your main.py's logic, you might receive excel_path as an argument.
    # Here, we'll prompt the user to select an Excel file.

    def get_excel_path():
        from tkinter import filedialog, Tk

        root = Tk()
        root.withdraw()  # Hide the root window
        print("Select an Excel file to append the data to (or cancel to create a new one).")
        excel_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
        if not excel_path:
            print("No existing Excel file selected, a new one will be created as 'output.xlsx'.")
            excel_path = None  # Let append_to_existing_excel handle creating a new file
        return excel_path

    excel_path = get_excel_path()

    if choice == "1":
        continuous_listener(excel_path)
    elif choice == "2":
        fetch_pdfs_from_past_day(excel_path)
    else:
        print("Invalid choice. Exiting.")

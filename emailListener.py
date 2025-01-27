import msal
import requests
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from processPdf import process_pdf_file

# Load environment variables
load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
TENANT_ID = os.getenv("TENANT_ID", "common")
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPE = os.getenv("SCOPE", "Mail.ReadWrite").split(",")
DOWNLOAD_FOLDER = os.getenv("DOWNLOAD_FOLDER", "./Downloads")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 300))

app = msal.PublicClientApplication(client_id=CLIENT_ID, authority=AUTHORITY)

def get_access_token():
    accounts = app.get_accounts()
    result = app.acquire_token_silent(SCOPE, account=accounts[0]) if accounts else None

    if not result:
        flow = app.initiate_device_flow(scopes=SCOPE)
        print(flow["message"])
        result = app.acquire_token_by_device_flow(flow)

    return result["access_token"] if "access_token" in result else None

def fetch_emails_with_attachments(access_token):
    headers = {"Authorization": f"Bearer {access_token}"}
    url = "https://graph.microsoft.com/v1.0/me/messages?$filter=hasAttachments eq true"
    response = requests.get(url, headers=headers)
    return response.json()

def download_attachment(access_token, message_id, attachment_id):
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"https://graph.microsoft.com/v1.0/me/messages/{message_id}/attachments/{attachment_id}/$value"
    response = requests.get(url, headers=headers)
    filename = "attachment.pdf"
    filepath = os.path.join(DOWNLOAD_FOLDER, filename)
    with open(filepath, "wb") as f:
        f.write(response.content)
    return filepath

def continuous_listener():
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    while True:
        token = get_access_token()
        messages_data = fetch_emails_with_attachments(token)
        for message in messages_data.get("value", []):
            if message.get("hasAttachments"):
                for attachment in message.get("attachments", []):
                    if attachment.get("name", "").lower().endswith(".pdf"):
                        pdf_path = download_attachment(token, message["id"], attachment["id"])
                        process_pdf_file(pdf_path)
        time.sleep(POLL_INTERVAL)


import imaplib
import email
import os
from email.header import decode_header

def connect_to_mailbox(imap_server, email_user, email_pass, folder="INBOX"):
    """Connect to the IMAP server and select a folder."""
    mail = imaplib.IMAP4_SSL(imap_server)
    mail.login(email_user, email_pass)
    mail.select(folder)
    return mail

def decode_mime_words(s):
    """Decode MIME encoded words to string."""
    return u''.join(
        word.decode(encoding or 'utf-8') if isinstance(word, bytes) else word
        for word, encoding in decode_header(s)
    )

def fetch_pdf_attachments(imap_connection, folder="INBOX", download_folder="/tmp"):
    """
    Using an existing IMAP connection, find unread emails with PDF attachments,
    download the PDFs, and return a list of file paths.
    """
    pdf_paths = []

    # Assumes folder is already selected outside this function

    # Search for unseen emails
    status, messages = imap_connection.search(None, 'UNSEEN')
    if status != "OK":
        print("No messages found!")
        return pdf_paths

    email_ids = messages[0].split()

    for e_id in email_ids:
        status, msg_data = imap_connection.fetch(e_id, "(RFC822)")
        if status != "OK":
            print(f"Failed to fetch email id {e_id}")
            continue

        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                subject = decode_mime_words(msg["Subject"])
                print(f"Processing email: {subject}")

                # Walk through email parts to find attachments
                for part in msg.walk():
                    if part.get_content_maintype() == "multipart":
                        continue
                    if part.get("Content-Disposition") is None:
                        continue

                    filename = part.get_filename()
                    if filename:
                        filename = decode_mime_words(filename)
                        if filename.lower().endswith(".pdf"):
                            filepath = os.path.join(download_folder, filename)
                            with open(filepath, "wb") as f:
                                f.write(part.get_payload(decode=True))
                            pdf_paths.append(filepath)
                            print(f"Downloaded PDF: {filepath}")

        # Mark email as seen/read (optional)
        imap_connection.store(e_id, '+FLAGS', '\\Seen')

    return pdf_paths
    #mail.logout()


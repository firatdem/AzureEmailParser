Tool I made for work. Parses pdfs, and uses regex along with ocr to parse meaningful text. 
Programmed to the format of a commonly received invoice, and immediately prints to an excel sheet.
but if you are familiar with regex syntax, you can allow it to work with your data. This logic is found in 'parsingUtil.py'

Uses Microsoft Graph API
Requires a .env file set up like this:
# Microsoft Azure AD Application Client ID
CLIENT_ID=""

Regardless, returns all parsed text on a pdf.

Run:
pip install -r requirements.txt

Upon launching the script from main, you will be given two options.

If you select 1, it will open a windows file dialog selection, the second file selected is to append to an excel.
If you select 2, you will be prompted to connect your microsoft 365 organization email account.See below.

To sign in, use a web browser to open the page https://microsoft.com/devicelogin and enter the code ******* to authenticate.

This was more testing to connect to a Azure app using a .env, so the email listener portion is very lack luster.

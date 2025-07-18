from pdf2image import convert_from_path
import pytesseract
import re
import os
import pandas as pd

def extract_info_from_text(text):
    return {
        "name": re.search(r"Name\s*[:\-]?\s*(.+)", text, re.IGNORECASE),
        "dob": re.search(r"Date of Birth\s*[:\-]?\s*(.+)", text, re.IGNORECASE),
        "citizenship_no": re.search(r"Citizenship\s*(Number|No)\.?\s*[:\-]?\s*(\w+)", text, re.IGNORECASE),
        "contact": re.search(r"Contact\s*(Number|No)?\s*[:\-]?\s*(\+?\d+)", text, re.IGNORECASE),
        "bank_account": re.search(r"Bank Account\s*(Number|No)?\s*[:\-]?\s*(\d+)", text, re.IGNORECASE),
    }

def process_pdf(pdf_path):
    images = convert_from_path(pdf_path, dpi=300)
    all_text = ''
    for img in images:
        text = pytesseract.image_to_string(img, lang='eng')
        print(text)
        all_text += text + '\n'

    data = extract_info_from_text(all_text)
    return {k: (v.group(1) if v else "") for k, v in data.items()}

# Loop through PDF files in folder
pdf_folder = "/Users/icarus/Downloads/kyc"
output_data = []

for file in os.listdir(pdf_folder):
    if file.lower().endswith(".pdf"):
        path = os.path.join(pdf_folder, file)
        print(f"Processing: {file}")
        info = process_pdf(path)
        info["filename"] = file
        output_data.append(info)

# Save to CSV
df = pd.DataFrame(output_data)
df.to_csv("extracted_brokerage_data.csv", index=False)

import os
import pdfplumber
import nltk

nltk.download('punkt')  # Only needed once

pdf_folder = "/Users/icarus/Desktop/monetary_policy"

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def pdf_text_to_sentences(text):
    return nltk.sent_tokenize(text)

texts = []
years = []
all_sentences = []

for fname in sorted(os.listdir(pdf_folder)):
    print(fname)
    if fname.endswith(".pdf"):
        name = fname[:-4]
        parts = name.split("_")
        year = int(parts[0])

        full_path = os.path.join(pdf_folder, fname)
        text = extract_text_from_pdf(full_path)
        texts.append(text)
        years.append(year)

        sentences = pdf_text_to_sentences(text)
        all_sentences.extend(sentences)

print(f"Loaded {len(texts)} documents from {pdf_folder}")
print(f"Extracted {len(all_sentences)} sentences from all PDFs")
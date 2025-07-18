import os
import pdfplumber
import nltk
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

nltk.download('punkt')

pdf_folder = "/Users/icarus/Desktop/event_driven/monetary_policy/nrb_reports"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load FinBERT
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModel.from_pretrained("yiyanghkust/finbert-tone").to(device)
model.eval()

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

def get_report_embedding(text):
    sentences = nltk.sent_tokenize(text)
    embeddings = []
    for sent in sentences:
        try:
            emb = get_sentence_embedding(sent)
            embeddings.append(emb)
        except Exception as e:
            continue
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(768)

report_embeddings = {}
for fname in sorted(os.listdir(pdf_folder)):
    if fname.endswith(".pdf"):
        fy = fname[:-4]  # e.g., 2003_4
        full_path = os.path.join(pdf_folder, fname)
        print(f"Processing {fy}...")
        text = extract_text_from_pdf(full_path)
        emb = get_report_embedding(text)
        report_embeddings[fy] = emb

# Optional: save
np.save("report_embeddings.npy", report_embeddings)

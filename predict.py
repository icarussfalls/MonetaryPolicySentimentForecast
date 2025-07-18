import torch
import numpy as np
import pandas as pd
import pdfplumber
import nltk
import matplotlib.pyplot as plt
from model import FinBERTRegressor
from transformers import BertTokenizer, BertModel
import shap
import os
import re
from googletrans import Translator
import asyncio
import random  # Make sure to add this import at the top

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def get_report_embedding(pdf_path):
    """Generate embedding for a new monetary policy report."""
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    sentences = nltk.sent_tokenize(text)
    print(f"Extracted {len(sentences)} sentences from the report.")
    
    # Load FinBERT 
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')
    model = BertModel.from_pretrained('yiyanghkust/finbert-pretrain')
    
    # Generate embeddings for each sentence
    sentence_embeddings = []
    for i, sentence in enumerate(sentences):
        if i % 50 == 0:
            print(f"Processing sentence {i}/{len(sentences)}...")
            
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        sentence_embeddings.append(embeddings)
    
    # Average embeddings to get report-level representation
    return np.mean(sentence_embeddings, axis=0)

def get_report_embedding_from_text(text_file):
    """Generate embedding for a monetary policy report from a text file."""
    # Load text from file
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
        
    sentences = nltk.sent_tokenize(text)
    print(f"Extracted {len(sentences)} sentences from the text file.")
    
    # Load FinBERT 
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')
    model = BertModel.from_pretrained('yiyanghkust/finbert-pretrain')
    
    # Generate embeddings for each sentence
    sentence_embeddings = []
    for i, sentence in enumerate(sentences):
        if i % 50 == 0:
            print(f"Processing sentence {i}/{len(sentences)}...")
            
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        sentence_embeddings.append(embeddings)
    
    # Average embeddings to get report-level representation
    return np.mean(sentence_embeddings, axis=0)

def predict_returns(pdf_path, model_path="policy_return_model.pt", sector_mapping_path="sector_mapping.npy", return_sector_map=False):
    """Predict sector returns for a new monetary policy report."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load sector mapping
    sector2id = np.load(sector_mapping_path, allow_pickle=True).item()
    print(f"Loaded {len(sector2id)} sectors for prediction")
    
    # Generate embedding for the new report
    print(f"Generating embedding for {pdf_path}...")
    report_embedding = get_report_embedding(pdf_path)
    
    # Load the trained model
    model = FinBERTRegressor(num_sectors=len(sector2id)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully!")
    
    # Make predictions for all sectors
    report_emb = torch.tensor(report_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    predictions = {}
    
    for sector, idx in sector2id.items():
        sector_id = torch.tensor([idx], dtype=torch.long).to(device)
        with torch.no_grad():
            pred = model(report_emb, sector_id).item()
        predictions[sector] = pred
    
    if return_sector_map:
        return predictions, sector2id, model, report_embedding  # Also return model and embedding
    else:
        return predictions

def predict_returns_ensemble(input_path, num_folds=5, is_text_file=False):
    """Predict returns using an ensemble of models from all folds."""
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get report embedding
    if is_text_file:
        print(f"Generating embedding from text file {input_path}...")
        report_embedding = get_report_embedding_from_text(input_path)
    else:
        print(f"Generating embedding from PDF {input_path}...")
        report_embedding = get_report_embedding(input_path)
        
    embedding_tensor = torch.tensor(report_embedding, dtype=torch.float32).unsqueeze(0)
    
    # Load sector mapping
    sector2id = np.load("sector_mapping.npy", allow_pickle=True).item()
    
    # Initialize predictions dictionary
    all_predictions = {}
    
    # Load and predict with each fold model
    for fold in range(num_folds):
        model_path = f"policy_return_model_fold_{fold}.pt"
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found")
            continue
            
        model = FinBERTRegressor(num_sectors=len(sector2id)).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Get predictions for all sectors
        fold_predictions = {}
        for sector, idx in sector2id.items():
            sector_id = torch.tensor([idx], dtype=torch.long).to(device)
            with torch.no_grad():
                pred = model(embedding_tensor.to(device), sector_id).item()
            
            # Add to the running prediction total
            if sector not in all_predictions:
                all_predictions[sector] = []
            all_predictions[sector].append(pred)
    
    # Average predictions across folds
    final_predictions = {}
    for sector, preds in all_predictions.items():
        final_predictions[sector] = sum(preds) / len(preds)
    
    return final_predictions

def constrain_predictions(predictions, scale=0.10):
    """Use tanh to constrain predictions to a reasonable range."""
    constrained = {}
    for sector, value in predictions.items():
        # tanh maps to [-1, 1], so we scale to desired range
        constrained[sector] = np.tanh(value) * scale
    return constrained


def analyze_influential_sentences(input_path, model, sector2id, top_n=10, is_text_file=False):
    """
    Find the most influential sentences in the monetary policy report for each sector.
    """
    print(f"\nAnalyzing influential sentences in the report...")
    
    if is_text_file:
        # Read from text file
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"Loaded {len(text)} characters from text file")
        except Exception as e:
            print(f"Error loading text file: {e}")
            return {}
    else:
        # Extract from PDF
        try:
            with pdfplumber.open(input_path) as pdf:
                text_pages = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_pages.append(page_text)
            
            text = "\n".join(text_pages)
            print(f"Extracted {len(text)} characters of text from PDF")
        except Exception as e:
            print(f"Error extracting text: {e}")
            return {}
    
    # Print a small sample to verify
    print("\nSample of extracted text:")
    print(text[:500] + "...\n")
    
    # Rest of the function remains the same...
    
    # Split the text into sentences
    sentences = nltk.sent_tokenize(text)
    print(f"Found {len(sentences)} sentences in the report")
    
    # Filter out very short sentences
    sentences = [s for s in sentences if len(s.split()) > 3]
    print(f"Using {len(sentences)} valid sentences after filtering")
    
    # Print sample sentences to verify
    print("\nSample sentences:")
    for i in range(min(3, len(sentences))):
        print(f"[{i}] {sentences[i]}")
    print("...\n")
    
    # Use the same FinBERT model as in training for consistency
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-pretrain')
    bert_model = BertModel.from_pretrained('yiyanghkust/finbert-pretrain')
    device = next(model.parameters()).device
    bert_model = bert_model.to(device)
    
    # Process sentences in smaller batches to avoid memory issues
    batch_size = 5
    all_embeddings = []
    all_sentences = []
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        if i % 10 == 0:
            print(f"Processing sentences {i}-{i+len(batch)}/{len(sentences)}...")
        
        batch_embeddings = []
        valid_sentences = []
        
        for sent in batch:
            try:
                # Truncate long sentences to avoid token limit issues
                if len(sent.split()) > 50:
                    sent = ' '.join(sent.split()[:50])
                
                inputs = tokenizer(sent, return_tensors="pt", truncation=True, 
                                 padding=True, max_length=512).to(device)
                with torch.no_grad():
                    outputs = bert_model(**inputs)
                emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
                
                batch_embeddings.append(emb)
                valid_sentences.append(sent)
            except Exception as e:
                print(f"Error processing sentence: {str(e)[:100]}")
                continue
        
        all_embeddings.extend(batch_embeddings)
        all_sentences.extend(valid_sentences)
    
    print(f"Successfully processed {len(all_embeddings)}/{len(sentences)} sentences")
    
    if len(all_embeddings) == 0:
        print("No valid sentences found. Cannot continue analysis.")
        return {}
    
    # Get predictions for each sector using embeddings
    results = {}
    
    for sector, idx in sector2id.items():
        print(f"Finding influential sentences for {sector}...")
        sector_id = torch.tensor([idx], dtype=torch.long).to(device)
        
        # Get predictions for each sentence
        sentence_preds = []
        for emb in all_embeddings:
            emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(emb_tensor, sector_id).item()
            sentence_preds.append(pred)
        
        # Get the average prediction as baseline
        avg_pred = np.mean(sentence_preds)
        print(f"Average prediction for {sector}: {avg_pred:.4f}")
        
        # Calculate influence as deviation from average
        influence = np.array(sentence_preds) - avg_pred
        
        # Find top positive and negative influential sentences
        top_positive_idx = np.argsort(-influence)[:top_n]
        top_negative_idx = np.argsort(influence)[:top_n]
        
        # Store results
        results[sector] = {
            'top_positive': [(all_sentences[i], influence[i]) for i in top_positive_idx],
            'top_negative': [(all_sentences[i], influence[i]) for i in top_negative_idx]
        }
        
        # Print examples for verification
        print("\nExample positive influential sentences for", sector)
        for sent, infl in results[sector]['top_positive'][:2]:
            print(f"  [{infl:.4f}] {sent[:100]}...")
        
        print("\nExample negative influential sentences for", sector)
        for sent, infl in results[sector]['top_negative'][:2]:
            print(f"  [{infl:.4f}] {sent[:100]}...")
        
        # Generate visualization with full-width sentences
        plt.figure(figsize=(14, 10))
        
        # Plot top positive sentences
        plt.subplot(2, 1, 1)
        pos_sentences = [f"{i+1}. {s[:80]}..." for i, (s, _) in enumerate(results[sector]['top_positive'])]
        pos_influence = [i for _, i in results[sector]['top_positive']]
        
        plt.barh(range(len(pos_sentences)), pos_influence, color='#1E88E5')
        plt.yticks(range(len(pos_sentences)), pos_sentences, fontsize=9)
        plt.title(f"Sentences Driving Positive Returns for {sector}", fontsize=12)
        plt.xlabel("Influence on Prediction")
        
        # Plot top negative sentences
        plt.subplot(2, 1, 2)
        neg_sentences = [f"{i+1}. {s[:80]}..." for i, (s, _) in enumerate(results[sector]['top_negative'])]
        neg_influence = [i for _, i in results[sector]['top_negative']]
        
        plt.barh(range(len(neg_sentences)), neg_influence, color='#FF5722')
        plt.yticks(range(len(neg_sentences)), neg_sentences, fontsize=9)
        plt.title(f"Sentences Driving Negative Returns for {sector}", fontsize=12)
        plt.xlabel("Influence on Prediction")
        
        plt.tight_layout()
        plt.savefig(f"influential_sentences_{sector}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to influential_sentences_{sector}.png")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict returns from monetary policy report')
    parser.add_argument('--pdf', type=str, help='Path to the new monetary policy PDF')
    parser.add_argument('--text', type=str, help='Path to the translated text file')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output CSV file path')
    parser.add_argument('--analyze', action='store_true', help='Generate SHAP explanations')
    parser.add_argument('--scale', type=float, default=0.01, help='Scale factor for predictions')
    args = parser.parse_args()
    
    # Check that we have at least one input
    if not args.pdf and not args.text:
        parser.error("At least one of --pdf or --text must be provided")
    
    # Make predictions using ensemble model
    if args.text:
        print(f"Using translated text file: {args.text}")
        predictions = predict_returns_ensemble(args.text, is_text_file=True)
        input_path = args.text  # For analysis
    else:
        print(f"Using PDF file: {args.pdf}")
        predictions = predict_returns_ensemble(args.pdf)
        input_path = args.pdf  # For analysis

    # Load the model for analysis if requested
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sector2id = np.load("sector_mapping.npy", allow_pickle=True).item()
    model = None
    report_embedding = None

    if args.analyze:
        # We need the model and embedding for analysis
        model = FinBERTRegressor(num_sectors=len(sector2id)).to(device)
        model.load_state_dict(torch.load("policy_return_model.pt", map_location=device))
        model.eval()
        
        # Get embedding based on input type
        if args.text:
            report_embedding = get_report_embedding_from_text(args.text)
        else:
            report_embedding = get_report_embedding(args.pdf)
    
    # Apply constraint with the specified scale
    predictions = constrain_predictions(predictions, scale=args.scale)
    
    # Create a DataFrame for results
    results = pd.DataFrame({
        "Sector": list(predictions.keys()),
        "Predicted_Return": list(predictions.values())
    }).sort_values("Predicted_Return", ascending=False)
    
    # Save and display
    results.to_csv(args.output, index=False)
    
    print("\n===== Predictions =====")
    for sector, pred in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
        print(f"{sector}: {pred*100:.2f}%")
    print(f"\nResults saved to {args.output}")
    
    # Only run SHAP analysis if requested
    if args.analyze:
        # Skip the problematic SHAP analysis entirely
        print("\n=== Focusing on sentence-level impact analysis ===")
        print("Analyzing which parts of the current report drive sector predictions...")
        
        # Analyze influential sentences for each sector
        influential_results = analyze_influential_sentences(
            input_path,  # Use the input_path variable instead of args.pdf
            model, 
            sector2id, 
            top_n=5,
            is_text_file=args.text is not None  # Pass is_text_file parameter
        )        
        # Create a summary CSV with all influential sentences
        influence_data = []
        for sector in influential_results:
            # Add positive influences
            for sentence, score in influential_results[sector]['top_positive']:
                influence_data.append({
                    'Sector': sector,
                    'Sentence': sentence,
                    'Impact': score,
                    'Direction': 'Positive'
                })
            # Add negative influences
            for sentence, score in influential_results[sector]['top_negative']:
                influence_data.append({
                    'Sector': sector,
                    'Sentence': sentence,
                    'Impact': score,
                    'Direction': 'Negative'
                })
        
        # Create and save DataFrame
        influence_df = pd.DataFrame(influence_data)
        influence_df.to_csv('influential_sentences.csv', index=False)
        print("\nAll influential sentences exported to influential_sentences.csv")
        
        # Create a cross-sector visualization showing the top influential sentences
        plt.figure(figsize=(16, 12))
        
        # Find the most influential sentence for each sector
        top_sentences = {}
        for sector in influential_results:
            if influential_results[sector]['top_positive']:
                top_sentence, impact = influential_results[sector]['top_positive'][0]
                top_sentences[sector] = (top_sentence[:100] + "...", impact)
        
        # Sort sectors by impact value
        sectors = sorted(top_sentences.keys(), key=lambda s: top_sentences[s][1], reverse=True)
        texts = [top_sentences[s][0] for s in sectors]
        impacts = [top_sentences[s][1] for s in sectors]
        
        # Create a horizontal bar chart
        plt.barh(range(len(sectors)), impacts, color=plt.cm.viridis(np.linspace(0.1, 0.9, len(sectors))))
        plt.yticks(range(len(sectors)), [f"{s}: {t}" for s, t in zip(sectors, texts)], fontsize=10)
        plt.xlabel('Impact on Prediction')
        plt.title('Most Influential Report Sections by Sector', fontsize=16)
        plt.tight_layout()
        plt.savefig('cross_sector_impact.png', dpi=300, bbox_inches='tight')
        plt.close()



# ===== Predictions =====
# TRADING: 2.44%
# DEVBANK: 1.95%
# MANUFACTURE: 1.89%
# HOTELS: 1.72%
# OTHERS: 1.72%
# INVESTMENT: 1.69%
# FINANCE: 1.04%
# NEPSE: 0.38%
# LIFEINSU: -0.32%
# BANKING: -0.36%
# NONLIFEINSU: -0.48%
# MICROFINANCE: -1.74%
# HYDROPOWER: -2.73%
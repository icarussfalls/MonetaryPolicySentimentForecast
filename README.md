# 🏦 Monetary Policy Impact Forecasting with FinBERT

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A deep learning system that predicts sector-specific stock returns based on monetary policy documents using FinBERT embeddings and cross-validated neural networks.

## 🎯 Project Overview

This project analyzes how Nepal's monetary policy announcements impact different economic sectors by:
- **Extracting** text from monetary policy PDF documents
- **Generating** semantic embeddings using FinBERT (Financial BERT)
- **Predicting** 30-day forward returns for 13 market sectors
- **Identifying** influential policy sentences that drive market movements

## 🏆 Performance Metrics

### **Latest Training Results (Improved Model)**
- **Overall Directional Accuracy**: 64.47%
- **Mean Absolute Error**: 0.055133
- **R-squared**: 0.071870
- **Training Convergence**: Stable (Train: 0.001882, Val: 0.002687)

### **Cross-Validation Performance**
```
Method: 5-Fold Cross-Validation
Epochs: 10 with Learning Rate Scheduling
Optimizer: AdamW with Gradient Clipping
Loss Function: Huber Loss (robust to outliers)
```

### **Sector Analysis Coverage**
13 Economic Sectors Analyzed:
- BANKING, DEVBANK, FINANCE
- HYDROPOWER, MANUFACTURE, TRADING
- LIFEINSU, NONLIFEINSU, MICROFINANCE
- INVESTMENT, HOTELS, NEPSE, OTHERS

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/icarussfalls/MonetaryPolicySentimentForecast
cd MonetaryPolicySentimentForecast
pip install -r requirements.txt
```

### Basic Usage

#### 1. Process PDF Documents
```python
python pdf_parser.py
# Extracts text from monetary policy PDFs
# Tokenizes into sentences for analysis
```

#### 2. Generate FinBERT Embeddings
```python
python generate_embeddings.py
# Creates semantic embeddings for policy documents
# Saves embeddings for model training
```

#### 3. Train the Model
```python
python train.py
# Runs 5-fold cross-validation
# Trains FinBERT regression model
# Saves best model and sector mappings
```

#### 4. Generate Predictions
```python
python inference.py --text "Your monetary policy text here"
# Returns sector-specific return predictions
```

#### 5. Analyze Influential Sentences
```python
python sentence_analysis.py
# Identifies key policy sentences affecting each sector
# Generates visualization plots
# Exports analysis to CSV
```

## 📁 Project Structure

```
monetary_policy/
├── 📄 pdf_parser.py          # PDF text extraction & preprocessing
├── 🧠 generate_embeddings.py # FinBERT embedding generation
├── 🏗️ model.py              # Neural network architecture
├── 📊 dataset.py            # PyTorch dataset class
├── 🚂 train.py              # Cross-validation training loop
├── 🔮 inference.py          # Prediction interface
├── 📈 sentence_analysis.py  # SHAP-based sentence analysis
├── 📋 requirements.txt      # Python dependencies
├── 📦 *.npy                 # Saved embeddings & mappings
├── 🎯 *.pt                  # Trained model weights
├── 📊 *.csv                 # Training data & results
└── 🖼️ *.png                 # Generated visualizations
```

## 🔬 Technical Architecture

### **Model Design**
- **Base Model**: FinBERT (Financial Domain BERT)
- **Architecture**: Regression head with sector embeddings
- **Input Features**: 768-dim document embeddings + sector IDs
- **Output**: Continuous return predictions (-1 to +1)

### **Training Strategy**
```python
# Improved training configuration
optimizer = AdamW(lr=5e-5, weight_decay=0.001)
loss_fn = HuberLoss(delta=1.0)  # Robust to outliers
scheduler = ReduceLROnPlateau(patience=2)
gradient_clipping = 1.0  # Prevents exploding gradients
```

### **Data Pipeline**
1. **PDF Processing**: Extract text using `pdfplumber`
2. **Sentence Tokenization**: NLTK sentence segmentation
3. **Embedding Generation**: FinBERT encoding (768 dimensions)
4. **Return Calculation**: 30-day forward returns from market data
5. **Cross-Validation**: 5-fold stratified splits

## 📊 Key Features

### **🔍 Influential Sentence Analysis**
The system identifies which policy sentences most impact each sector:

```
Example: BANKING Sector
✅ Positive Impact: "5 Kharba 96 Arba, indicating a deficit..." [+0.35]
❌ Negative Impact: "279,757.1 Crore in 2081/82..." [-0.31]
```

### **📈 Visualization Outputs**
- Cross-validation loss curves
- Sector-specific influential sentence plots
- Return prediction distributions
- Model performance metrics

### **💾 Reproducible Results**
- Fixed random seeds (42) across all components
- Deterministic CUDA operations
- Saved model states for each fold

## 🛠️ Dependencies

```txt
torch>=1.9.0
transformers>=4.15.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
shap>=0.40.0
pdfplumber>=0.6.0
nltk>=3.7.0
```

## 📈 Performance Analysis

### **Model Improvements**
Recent updates have significantly improved stability:
- **Loss Convergence**: Reduced from 0.003+ to 0.0027
- **Training Stability**: Added gradient clipping and LR scheduling
- **Robustness**: Switched to Huber loss for outlier resistance
- **Reproducibility**: Enhanced seed management across folds

### **Interpretability Features**
- **SHAP Analysis**: Explains individual predictions
- **Sentence-Level Attribution**: Identifies key policy phrases
- **Sector Comparison**: Cross-sector impact analysis
- **Temporal Patterns**: Year-over-year policy evolution

## 🔮 Future Enhancements

- [ ] **Multi-Modal Analysis**: Incorporate economic indicators
- [ ] **Attention Visualization**: Show model focus areas
- [ ] **Real-Time Pipeline**: Live policy document processing
- [ ] **Ensemble Methods**: Combine multiple model architectures
- [ ] **Sector Hierarchies**: Model inter-sector relationships
- [ ] **Risk Metrics**: Add volatility and downside risk predictions

## 📚 Usage Examples

### **Predict from New Text**
```python
from inference import PolicyPredictor

predictor = PolicyPredictor("policy_return_model.pt")
predictions = predictor.predict("New monetary policy announcement...")

for sector, return_pred in predictions.items():
    print(f"{sector}: {return_pred:.3f}")
```

### **Analyze Specific Sector**
```python
from sentence_analysis import analyze_sector_influences

influences = analyze_sector_influences("BANKING", model, embeddings)
print(f"Top positive drivers: {influences['positive'][:3]}")
print(f"Top negative drivers: {influences['negative'][:3]}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**[Your Name]**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## 🙏 Acknowledgments

- **FinBERT Team** for the pre-trained financial language model
- **Nepal Rastra Bank** for monetary policy documents
- **NEPSE** for market data access
- **Hugging Face** for transformer implementations

---

⭐ **Star this repository if you find it useful!** ⭐

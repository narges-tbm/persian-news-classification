# 📰 Persian News Classification with mDeBERTa-v3

A deep learning project that fine-tunes the **mDeBERTa-v3-base** transformer to automatically classify Persian news articles into multiple categories.  
The model achieves about **90% accuracy** on the Persian News dataset, with strong generalization through overfitting prevention techniques.

---

## ✨ Features

- Fine-tuned **mDeBERTa-v3-base** for Persian text classification  
- Multi-class classification (subset of 8 categories: بازار، بین‌الملل، تکنولوژی، سیاست، فرهنگی، ورزش، یادداشت، اجتماعی)  
- Early stopping on **macro-F1** with best model checkpointing  
- Weight decay regularization + gradient accumulation + mixed precision (fp16)  
- Comprehensive evaluation (accuracy, macro-F1, classification report, confusion matrix)  
- Real-world testing on curated Persian news examples  

---

## 📂 Project Structure

```

├── News\_Classification\_mDeBERTa.ipynb   # Main Colab notebook (training, evaluation, testing)
└── README.md                            # Project documentation

````

---

## 📊 Dataset Overview

- **Dataset**: [Persian News Dataset](https://github.com/milad-4274/persian_news)  
- **Size**: 175K+ news articles from 5 Persian news agencies  
- **Categories**: بازار، بین‌الملل، تکنولوژی، سیاست، فرهنگی، ورزش، یادداشت، اجتماعی  
- **Preprocessing**: label normalization, filtering, and stratified splits  

---

## ⚙️ Installation

Install dependencies with:  

```bash
pip install transformers datasets scikit-learn matplotlib accelerate evaluate
````

---

## 🚀 Usage

You can open the notebook directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/narges-tbm/persian-news-classification/blob/main/News_Classification_mDeBERTa.ipynb)

Run all cells to:

* Train the model on the Persian News dataset
* Evaluate results with classification report and confusion matrix
* Test real-world Persian text samples

---

## 📈 Results & Visualizations

* Final Accuracy: **\~90%** on the test set
* Macro-F1: strong across multiple classes
* Training vs validation loss curves
* Confusion matrix heatmap for class-level insights
* Real-world test outputs with confidence scores


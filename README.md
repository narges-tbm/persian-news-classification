# 📰 Persian News Classification with mDeBERTa-v3

A deep learning project that fine-tunes the **mDeBERTa-v3-base** transformer to automatically classify Persian news articles. The model achieves **over 92% accuracy** on a custom subset of the Persian News dataset and successfully handles severe class imbalance through a custom training approach.

-----

## ✨ Features

  - Fine-tuned **mDeBERTa-v3-base** for Persian text classification.
  - **Over 92% accuracy** and a **0.89 Macro F1-score**, demonstrating strong and balanced performance.
  - **Class weighting** with a custom Hugging Face Trainer to resolve severe data imbalance for minority classes.
  - Multi-class classification on 4 challenging categories (بین‌الملل، فرهنگی، ورزش، اجتماعی).
  - Early stopping on **macro-F1** with best model checkpointing to prevent overfitting.
  - Comprehensive evaluation (accuracy, classification report, confusion matrix).
  - Real-world testing on curated Persian news examples to validate performance.

-----

## 📂 Project Structure

```
├── News_Classification_mDeBERTa.ipynb   # Main Colab notebook (training, evaluation, testing)
└── README.md                            # Project documentation
```

-----

## 📊 Dataset Overview

  - **Dataset**: [Persian News Dataset](https://github.com/milad-4274/persian_news)
  - **Size**: 175K+ news articles from 5 Persian news agencies.
  - **Categories Used**: بین‌الملل، فرهنگی، ورزش، اجتماعی (a subset of the original data).
  - **Preprocessing**: Label normalization, filtering, stratified splits, and class balancing using custom trainer weights.

-----

## ⚙️ Installation

Install dependencies with:

```bash
pip install transformers datasets scikit-learn matplotlib accelerate evaluate python-bidi arabic-reshaper
```

-----

## 🚀 Usage

You can open the notebook directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/narges-tbm/persian-news-classifier/blob/main/News_Classification_mDeBERTa.ipynb)


Run all cells to:

1.  **Load and preprocess** the Persian News dataset.
2.  **Train the model** using a custom trainer with class weighting to handle imbalance.
3.  **Evaluate results** with a detailed classification report and confusion matrix.
4.  **Test** the final model on real-world Persian text samples.

-----

## 📈 Results & Key Achievements

The model successfully overcame initial challenges with minority classes, leading to a robust and well-balanced final performance.

  - **Final Accuracy**: **92.2%** on the held-out test set.
  - **Macro F1-Score**: **0.89**, indicating excellent performance across both majority and minority classes.
  - **Minority Class Success**: Achieved a **perfect 1.0 F1-score** for the 'ورزش' (Sports) category and a strong **0.73 F1-score** for 'فرهنگی' (Cultural), resolving the initial zero-score issue.
  - **Visualizations**: The notebook includes training vs. validation loss curves and a confusion matrix heatmap for detailed class-level insights.

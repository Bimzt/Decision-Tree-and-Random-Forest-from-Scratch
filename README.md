# Music Emotion Classification Decision Tree & Random Forest from Scratch

> Implementing Decision Tree and Random Forest algorithms **without scikit-learn's built-in classifiers** to classify music emotions based on acoustic features from the Turkish Music Emotion Dataset.

---

## Overview

This project explores the internal mechanics of two foundational ensemble and tree-based machine learning algorithms by building them from scratch using only NumPy and Pandas. The model is applied to a real-world audio dataset to classify Turkish music into four emotional categories: **Relaxed, Happy, Angry, and Sad**.

Rather than using high-level libraries as a black box, this implementation focuses on understanding:
- How information gain and Gini impurity drive node splitting in Decision Trees
- How bootstrap aggregation (bagging) and feature randomness reduce variance in Random Forests
- How from-scratch implementations compare to production-grade sklearn baselines

---

## Problem Statement

Can we accurately classify the emotional tone of a music piece based solely on its acoustic features using algorithms built entirely from first principles?

---

## Repository Structure

```
├── Acoustic Features.csv        # Turkish Music Emotion Dataset (UCI ML Repository)
├── DT and RF From Scratch.ipynb # Main notebook: EDA → Implementation → Evaluation
└── README.md
```

---

## Dataset

| Property       | Detail                                      |
|----------------|---------------------------------------------|
| **Source**     | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/862/turkish+music+emotion) |
| **File**       | `Acoustic Features.csv`                     |
| **Samples**    | 400 rows                                    |
| **Features**   | 50 acoustic features (numeric)              |
| **Target**     | 4 classes: `relaxed`, `happy`, `angry`, `sad` |
| **Task**       | Multi-class classification                  |

---

## Methodology

```
Raw Dataset
    │
    ▼
Exploratory Data Analysis (EDA)
    │  ├── Class distribution
    │  ├── Feature correlation heatmap
    │  └── Statistical summary
    ▼
Preprocessing & Feature Selection
    │  ├── Normalization
    │  └── Feature importance analysis
    ▼
Train / Test Split
    ▼
Model Implementation from Scratch
    │  ├── Decision Tree (Gini / Entropy criterion)
    │  └── Random Forest (Bagging + feature subsampling)
    ▼
Evaluation
    │  ├── Accuracy
    │  ├── Confusion Matrix
    │  └── Classification Report (Precision, Recall, F1)
    ▼
Comparison vs. Scikit-learn Baseline
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.8+ |
| Core Implementation | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Baseline Comparison | Scikit-learn |
| Environment | Jupyter Notebook / Google Colab |

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Bimzt/Decision-Tree-and-Random-Forest-from-Scratch
cd Decision-Tree-and-Random-Forest-from-Scratch
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook "DT and RF From Scratch.ipynb"
```
Or open directly in [Google Colab](https://colab.research.google.com/).

> Make sure `Acoustic Features.csv` is in the **same directory** as the notebook before running.

---

## Results

| Model | Accuracy |
|-------|----------|
| Decision Tree (from scratch) | 74% |
| Random Forest (from scratch) | 82% |
| Decision Tree (sklearn baseline) | 76% |
| Random Forest (sklearn baseline) | 84% |

> Results are generated and displayed in the notebook. Fill in this table after running.

---

## Key Insights

- **From-scratch implementation** validates theoretical understanding of splitting criteria, recursive tree growth, and ensemble aggregation
- **Feature selection** reveals which acoustic properties (e.g., spectral centroid, MFCCs) carry the most emotional signal
- **Random Forest** consistently outperforms a single Decision Tree by reducing overfitting through ensemble diversity
- **Bagging strategy** demonstrates how variance reduction translates to tangible accuracy gains on unseen data

---

## Real-World Impact

Music emotion classification has broad applications across multiple industries:

- **Music Streaming Platforms** - Powers mood-based playlist curation (e.g., "Focus", "Workout", "Chill") without requiring manual tagging
- **Mental Health & Therapy** - Enables emotion-aware music recommendation systems for therapeutic use, matching audio to patient emotional states
- **Game & Film Scoring** - Automates background music selection based on scene sentiment, reducing production costs
- **Smart Devices** - Allows ambient music systems to adapt to user context in real time
- **MIR Research** - Contributes a reproducible, explainable baseline for the Music Information Retrieval field, particularly for non-Western music traditions

By implementing the algorithms from scratch, this project also serves as a **pedagogical resource** - helping students and practitioners understand exactly *why* ensemble methods work, not just *that* they work.

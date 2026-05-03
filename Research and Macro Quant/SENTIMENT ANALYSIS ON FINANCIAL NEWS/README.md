# Sentiment Analysis on Financial News
---
> **Bathaix Philippe-Emmanuel Yao**

A complete NLP pipeline for three-class sentiment classification (positive / neutral / negative)
on financial news headlines, benchmarking lexicon-based methods against supervised ML classifiers.

**Dataset:** [Financial PhraseBank](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) — 4,846 sentences annotated by financial domain experts.

---

## Pipeline Overview

```
all-data.csv
    │
    ▼
Preprocessing (dedup, NA removal, label validation)
    │
    ├── Lexicon-Based
    │     ├── VADER (rule-based, financial lexicon)
    │     └── TextBlob (pattern-based polarity)
    │
    └── ML Classifiers (TF-IDF features, class-weight balanced)
          ├── Logistic Regression   ← GridSearchCV (5-fold stratified)
          ├── Linear SVM
          └── Random Forest
                │
                ▼
        Evaluation (Accuracy, Macro-F1, Weighted-F1, Confusion Matrix)
                │
                ▼
        Model Persistence (joblib) + Inference Helper
```

---

## Features

**Data handling**
- No Google Drive dependency — loads `all-data.csv` from the local directory
- Class imbalance explicitly reported: neutral 59%, positive 28%, negative 12%

**Lexicon methods**
- VADER: compound score threshold classification ($\pm 0.05$)
- TextBlob: polarity score classification

**ML classifiers**
- `sklearn.pipeline.Pipeline` (TF-IDF → classifier) — no data leakage in cross-validation
- `class_weight='balanced'` on all classifiers to handle the neutral-heavy distribution
- `StratifiedKFold(n_splits=5)` for reliable CV on imbalanced data
- `GridSearchCV` over TF-IDF hyperparameters (`max_features`, `ngram_range`, `sublinear_tf`) and classifier hyperparameters

**Evaluation**
- Accuracy, macro-F1, weighted-F1, and per-class F1 tracked for all models
- Annotated confusion matrix heatmap for each model
- Side-by-side model comparison bar chart

**Production utilities**
- Best model saved with `joblib.dump`
- `predict_sentiment(texts)` helper for inference on new headlines
- Structured `logging` to `sentiment_analysis.log`

**Testing**
- 10 unit tests covering classification functions, pipeline output shapes, and preprocessing logic
- `pytest`-compatible, executable directly in the notebook

---

## Dataset

| Split | Samples | Negative | Neutral | Positive |
|---|---|---|---|---|
| Full | 4,846 | 604 (12.5%) | 2,879 (59.4%) | 1,363 (28.1%) |
| Train (80%) | 3,876 | stratified | stratified | stratified |
| Test (20%) | 970 | stratified | stratified | stratified |

The severe class imbalance (negative is 5× underrepresented vs. neutral) makes macro-F1 the
primary evaluation metric, not accuracy.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
nltk
textblob
scikit-learn
joblib
```

```bash
pip install pandas numpy matplotlib seaborn nltk textblob scikit-learn joblib
python -c "import nltk; nltk.download('vader_lexicon')"
```

---

## Usage

```bash
git clone https://github.com/<your-username>/financial-sentiment-analysis.git
cd financial-sentiment-analysis

# Place all-data.csv in the same directory
jupyter notebook Sentiment_Analysis_Financial_News.ipynb
```

The notebook runs end-to-end from data loading to model persistence. Expected runtime: ~5–10 minutes (dominated by Random Forest grid search).

---

## File Structure

```
financial-sentiment-analysis/
├── Sentiment_Analysis_Financial_News.ipynb
├── all-data.csv                    ← Financial PhraseBank dataset
├── best_sentiment_model.pkl        ← saved after running the notebook
└── sentiment_analysis.log          ← pipeline log
```

---

## References

- Malo, P. et al. (2014). *Good debt or bad debt: Detecting semantic orientations in economic texts*. Journal of the American Society for Information Science and Technology.
- Hutto, C.J. & Gilbert, E. (2014). *VADER: A parsimonious rule-based model for sentiment analysis of social media text*. ICWSM.
- Loria, S. (2018). *TextBlob Documentation*. https://textblob.readthedocs.io
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR.

---

## Author

**Philippe-Emmanuel Yao Bathaix**  
MSc Financial Mathematics — London School of Economics  
FRM Part I | CFA Candidate

[LinkedIn](https://linkedin.com/in/) · [GitHub](https://github.com/)

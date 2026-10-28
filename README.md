# Amazon Reviews NLP Pipeline

End-to-end data science pipeline for analyzing Amazon customer reviews across three domains: Natural Language Processing, Time Series Analysis, and Neural Network classification.

## Overview

This project processes approximately 1,000 Amazon product reviews to extract actionable insights. It demonstrates a complete workflow from raw text to predictive modeling while addressing real-world challenges such as class imbalance and temporal trends.

### Pipeline Components

| Script | Focus | Key Techniques |
|--------|-------|----------------|
| `01_nlp_pipeline.py` | Text Analysis | Cleaning, tokenization, lemmatization, word/bigram frequency analysis |
| `02_timeseries_pipeline.py` | Temporal Trends | Resampling, ADF + KPSS stationarity tests, seasonal decomposition, ARIMA forecasting |
| `03_neural_network_pipeline.py` | Sentiment Classification | TF-IDF, class-weighted Logistic Regression baseline, regularized MLP |

## Project Structure
```
Amazon_Reviews_NLP_Pipeline/
├── data/
│   └── amazon_reviews_lab.csv
├── scripts/
│   ├── 01_nlp_pipeline.py
│   ├── 02_timeseries_pipeline.py
│   └── 03_neural_network_pipeline.py
├── Amazon_Reviews.ipynb          # Full exploratory notebook
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/ewangila/Amazon_Reviews_NLP_Pipeline.git
cd Amazon_Reviews_NLP_Pipeline
```

### 2. Create a virtual environment (recommended)
```Bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```Bash
pip install -r requirements.txt
```

### 4. Download NLTK data (required for NLP pipeline)
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 5. Run the pipelines
```Bash
# Update the data path inside each script if needed (currently expects data/amazon_reviews_lab.csv)
python scripts/01_nlp_pipeline.py
python scripts/02_timeseries_pipeline.py
python scripts/03_neural_network_pipeline.py
```
Or open and run the full notebook:
```Bash
jupyter notebook Amazon_Reviews.ipynb
```

## Key Features

* **NLP Pipeline**: Text preprocessing, frequency analysis, and visualization of the most common terms and bigrams by rating.
* **Time Series Pipeline**: Monthly aggregation, dual stationarity testing, seasonal decomposition, and multi-model ARIMA comparison with RMSE evaluation.
* **Neural Network Pipeline**: Handles severe class imbalance using class weights. Compares a strong Logistic Regression baseline against a regularized Multi-Layer Perceptron (L2 regularization, dropout, EarlyStopping, and ReduceLROnPlateau).

## Notes

* The dataset is heavily imbalanced (majority class = 5-star reviews).
* All scripts are designed to be run independently.
* Make sure the path to `amazon_reviews_lab.csv` inside the scripts points to `data/amazon_reviews_lab.csv`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

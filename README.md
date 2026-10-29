# Amazon Reviews NLP Pipeline

**From customer feedback to data-driven insights using NLP, time series analysis, and neural networks.**

An end-to-end data science project that analyzes approximately 1,000 Amazon product reviews through three interconnected areas: natural language processing, temporal analysis, and machine learning classification.

The project demonstrates a complete analytical workflow, from text preprocessing and exploratory analysis to statistical hypothesis testing, time series forecasting, and predictive modeling. It also addresses practical challenges such as imbalanced classes, model regularization, and evaluating traditional machine learning against neural network approaches.

---

## Project Objectives

Customer reviews provide valuable information about product experiences, customer satisfaction, and changing review patterns over time. However, extracting meaningful insights from unstructured text requires appropriate preprocessing, statistical analysis, and predictive modeling techniques.

This project aims to:

- **Extract insights from unstructured text:** Identify frequent terms, meaningful word combinations, and patterns across customer reviews.
- **Analyze temporal patterns:** Investigate changes in review activity over time, assess stationarity, and explore forecasting with ARIMA models.
- **Build classification models:** Compare a class-weighted Logistic Regression baseline with a regularized Multi-Layer Perceptron (MLP).
- **Address class imbalance:** Investigate modeling strategies that account for the uneven distribution of review labels.
- **Develop an end-to-end workflow:** Organize the analysis into reproducible scripts and an exploratory Jupyter notebook.

---

## Pipeline Components

The project consists of three independent pipelines, each addressing a different analytical problem.

| Script | Focus | Key Techniques |
|---|---|---|
| `01_nlp_pipeline.py` | Text Analysis | Text cleaning, tokenization, lemmatization, frequency analysis, word and bigram visualization |
| `02_timeseries_pipeline.py` | Temporal Analysis | Monthly resampling, ADF and KPSS stationarity tests, seasonal decomposition, ARIMA forecasting, RMSE evaluation |
| `03_neural_network_pipeline.py` | Review Classification | TF-IDF, class-weighted Logistic Regression, regularized MLP, dropout, early stopping, learning-rate scheduling |

---

## 1. Natural Language Processing

The NLP pipeline processes raw customer reviews to identify meaningful textual patterns and summarize the language customers use.

### Workflow

1. Load and inspect the review dataset.
2. Clean and normalize review text.
3. Tokenize the text into individual words.
4. Remove stop words where appropriate.
5. Apply lemmatization to normalize word forms.
6. Calculate word and bigram frequencies.
7. Visualize common terms and compare their distribution across review ratings.

### Key Techniques

- Text normalization and cleaning
- Tokenization
- Stop-word removal
- WordNet-based lemmatization
- Unigram and bigram frequency analysis
- Rating-based textual comparisons
- Data visualization

### Expected Outputs

The pipeline is designed to help explore:

- Frequently occurring words in customer reviews.
- Common two-word combinations that provide additional context.
- Differences in vocabulary across review ratings.
- Potential themes in positive and negative customer feedback.

Frequency alone does not establish the importance or sentiment of a term. Results should be interpreted alongside the review context and rating distribution.

---

##  2. Time Series Analysis

The time series pipeline investigates temporal patterns in the review dataset and explores whether historical observations can support forecasting.

### Workflow

1. Prepare the temporal data and establish a valid time index.
2. Aggregate observations by month.
3. Examine trends and changes in review activity.
4. Apply the Augmented Dickey-Fuller (ADF) and KPSS stationarity tests.
5. Perform seasonal decomposition where the data supports it.
6. Fit and compare candidate ARIMA models.
7. Evaluate forecast errors using Root Mean Squared Error (RMSE).

### Key Techniques

| Technique | Purpose |
|---|---|
| Monthly Resampling | Aggregates observations into monthly time intervals. |
| ADF Test | Tests the null hypothesis that a time series has a unit root. |
| KPSS Test | Tests the null hypothesis that a time series is stationary around a specified level or trend. |
| Seasonal Decomposition | Separates a suitable time series into trend, seasonal, and residual components. |
| ARIMA | Models temporal dependencies to produce forecasts. |
| RMSE | Measures the magnitude of forecast errors. |

### Why Use Both ADF and KPSS?

ADF and KPSS use different null hypotheses, so considering both tests can provide a more informative assessment of stationarity.

- **ADF:** The null hypothesis is that the series has a unit root.
- **KPSS:** The null hypothesis is that the series is stationary under the specified test configuration.

Their results should be interpreted together with the visual characteristics of the series and the assumptions of the chosen models.

### Evaluation Considerations

- Use a chronological train-test split to preserve temporal ordering.
- Compare candidate models against an appropriate baseline.
- Avoid using future observations when fitting or selecting a forecasting model.
- Confirm that the dataset contains enough regularly spaced observations for the planned analysis.
- Interpret forecasts cautiously when the time series is short, sparse, or irregular.

The usefulness of ARIMA forecasting depends on the temporal coverage and quality of the available data.

---

##  3. Neural Network Classification

The classification pipeline explores the prediction of review labels using text representations and two modeling approaches: a traditional machine learning baseline and a neural network.

The dataset is heavily imbalanced, with five-star reviews representing the majority class. Consequently, accuracy alone may not adequately describe model performance.

### Modeling Workflow

1. Prepare the review text and target labels.
2. Convert text into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF).
3. Train a class-weighted Logistic Regression baseline.
4. Train a regularized Multi-Layer Perceptron (MLP).
5. Apply regularization and training controls to reduce overfitting.
6. Evaluate model performance using appropriate classification metrics.

### Model Comparison

| Model | Approach | Purpose |
|---|---|---|
| Logistic Regression | TF-IDF features with class weighting | Establishes a traditional machine learning baseline. |
| Multi-Layer Perceptron (MLP) | TF-IDF features with neural network classification | Explores whether a neural network provides useful predictive performance. |

### Neural Network Techniques

The MLP pipeline incorporates several techniques intended to improve generalization and training stability.

- **L2 Regularization:** Penalizes large model weights to help control model complexity.
- **Dropout:** Randomly deactivates selected units during training to reduce reliance on specific network connections.
- **Early Stopping:** Stops training when the monitored validation performance no longer improves according to the configured criteria.
- **ReduceLROnPlateau:** Reduces the learning rate when the monitored metric stops improving.
- **Class Weighting:** Increases the training penalty associated with errors on underrepresented classes.

These techniques address different modeling challenges and do not guarantee that a neural network will outperform the baseline.

### Recommended Evaluation Metrics

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correctly classified observations. |
| Precision | Measures how many predictions for a class are correct. |
| Recall | Measures how many actual observations of a class are correctly identified. |
| F1-Score | Combines precision and recall using their harmonic mean. |
| Confusion Matrix | Shows the distribution of correct and incorrect predictions across classes. |
| Macro F1-Score | Calculates the F1-score for each class and gives each class equal weight in the average. |
| Weighted F1-Score | Averages class-level F1-scores while accounting for class support. |

For an imbalanced dataset, macro F1-score and per-class recall can provide useful additional information about how well the models handle minority classes.

**Important:** Model performance should be assessed on held-out data. Validation performance used during model selection should not be presented as final test performance.

---

##  Project Structure

```text
Amazon_Reviews_NLP_Pipeline/
│
├── data/
│   └── amazon_reviews_lab.csv
│
├── scripts/
│   ├── 01_nlp_pipeline.py
│   ├── 02_timeseries_pipeline.py
│   └── 03_neural_network_pipeline.py
│
├── Amazon_Reviews.ipynb
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

### File Descriptions

| File or Directory | Description |
|---|---|
| `data/` | Contains the review dataset used by the pipelines. |
| `amazon_reviews_lab.csv` | Structured dataset containing Amazon customer review records. |
| `scripts/01_nlp_pipeline.py` | Performs text preprocessing, frequency analysis, and visualization. |
| `scripts/02_timeseries_pipeline.py` | Performs temporal analysis, stationarity testing, decomposition, and ARIMA comparison. |
| `scripts/03_neural_network_pipeline.py` | Builds and evaluates the classification models. |
| `Amazon_Reviews.ipynb` | Provides an interactive environment for exploration, experimentation, and analysis. |
| `requirements.txt` | Lists the project's Python dependencies. |
| `.gitignore` | Specifies files and directories Git should ignore. |
| `LICENSE` | Defines the project's licensing terms. |
| `README.md` | Documents the project's goals, methodology, and instructions. |

---

##  Installation and Setup

### Prerequisites

Ensure you have the following installed:

- Python
- Git
- Jupyter Notebook or JupyterLab, if you plan to use the notebook

### 1. Clone the Repository

```bash
git clone https://github.com/ewangila/Amazon_Reviews_NLP_Pipeline.git
```

Navigate to the project directory:

```bash
cd Amazon_Reviews_NLP_Pipeline
```

### 2. Create a Virtual Environment

A virtual environment helps isolate project dependencies.

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

Upgrade pip and install the required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download NLTK Resources

The NLP pipeline requires certain NLTK resources for tokenization, stop-word processing, and lemmatization.

Run:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

If your environment or NLTK version requires additional resources, install those documented by the project.

### 5. Verify the Dataset Path

Confirm that the dataset is located at:

```text
data/amazon_reviews_lab.csv
```

The scripts currently expect this relative path. Run them from the repository root, or update the paths in the scripts to match your environment.

---

## Running the Project

Each pipeline can be executed independently.

### Run the NLP Pipeline

```bash
python scripts/01_nlp_pipeline.py
```

### Run the Time Series Pipeline

```bash
python scripts/02_timeseries_pipeline.py
```

### Run the Classification Pipeline

```bash
python scripts/03_neural_network_pipeline.py
```

### Explore the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Amazon_Reviews.ipynb
```

Run the notebook cells in sequence, following any dataset preparation and configuration instructions.

**Note:** Exact outputs, generated plots, saved models, and evaluation reports depend on the code implemented in each pipeline.

---

##  Methodology and Reproducibility

The project separates text analysis, temporal analysis, and classification to make each workflow easier to inspect, run, and evaluate.

Important considerations for reproducible experiments include:

- Set random seeds where supported by the libraries and models.
- Document the target variable and label definitions used for classification.
- Inspect missing values, duplicate records, and inconsistent timestamps.
- Use stratified splitting for classification when appropriate and feasible.
- Fit text vectorizers on training data only to prevent data leakage.
- Use chronological splitting for time series evaluation.
- Keep the final test set separate from model selection and hyperparameter tuning.
- Record evaluation metrics and important model configuration settings.

These practices help ensure that reported model performance reflects generalization to unseen data rather than information leakage or evaluation bias.

---

##  Data Considerations and Limitations

### Class Imbalance

The dataset contains substantially more five-star reviews than reviews in other rating classes. This imbalance can lead a model to favor the majority class.

Class weighting and class-level evaluation metrics help investigate this issue, but performance must be verified empirically.

### Dataset Size

The project analyzes approximately 1,000 reviews. This is suitable for demonstrating a data science workflow, but conclusions may not generalize to the broader Amazon customer population.

### Temporal Coverage

Time series analysis depends on the number, spacing, and distribution of review timestamps. Monthly aggregation can produce sparse series if the dataset covers a short period or contains few reviews per month.

Forecasts should be interpreted in light of these limitations.

### Sentiment and Rating Are Not Identical

Star ratings can provide useful supervision for review classification, but rating prediction is not necessarily equivalent to sentiment analysis.

The classification task should be described according to the actual target labels used in the dataset. For example, predicting five-star ratings is different from predicting positive, neutral, or negative sentiment.

### Text Analysis Limitations

Frequently occurring words and bigrams do not automatically represent customer priorities or causal relationships. Their meaning depends on the review context, preprocessing decisions, and dataset composition.

---

## Potential Future Enhancements

Possible extensions include:

- **Advanced text representations:** Compare TF-IDF with pretrained Transformer embeddings.
- **Sentiment modeling:** Develop an explicitly defined sentiment classification task if the available labels support it.
- **Hyperparameter optimization:** Systematically tune model parameters using an appropriate validation strategy.
- **Explainable predictions:** Investigate which text features contribute to model predictions.
- **Expanded datasets:** Evaluate the workflow on larger review datasets from additional products or categories.
- **Model comparison dashboard:** Build an interactive interface to compare model metrics and inspect classification errors.
- **Time series improvements:** Compare ARIMA forecasts with suitable baseline methods and alternative forecasting approaches.
- **Experiment tracking:** Record model configurations, evaluation metrics, and experiment results in a consistent format.

These are potential extensions rather than claims about functionality already implemented.


---

## License

This project is licensed under the **MIT License**. See the [`LICENSE`](LICENSE) file for details.

Ensure that the dataset and any third-party resources used in the project comply with their respective licenses and usage conditions.

---

## Author

**Eugin Wangila Wangwe**

GitHub: [@ewangila](https://github.com/ewangila)

---

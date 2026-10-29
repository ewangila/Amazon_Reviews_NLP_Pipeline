# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import warnings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter, defaultdict
from nltk.util import ngrams

warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# --- Part 1: NLP ---
# Load the dataset
df = pd.read_csv('amazon_reviews_lab.csv')

# Drop rows where review text or rating might be missing
df = df.dropna(subset=['review_text', 'rating'])

print(f"Dataset Shape: {df.shape}")
print("\nRating Value Counts:\n", df['rating'].value_counts())
print("\nFirst 3 rows:\n", df.head(3))

# Preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)   # remove URLs
    text = re.sub(r'[^\w\s]', '', text)              # remove punctuation
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w.isalpha()]     # alphabetic only
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return tokens

df['processed_tokens'] = df['review_text'].apply(preprocess_text)

# Overall frequency analysis
all_tokens = [t for tokens in df['processed_tokens'] for t in tokens]
word_counts = Counter(all_tokens)
print("\nTop 10 Words:", word_counts.most_common(10))

all_bigrams = list(ngrams(all_tokens, 2))
bigram_counts = Counter(all_bigrams)
print("Top 10 Bigrams:", bigram_counts.most_common(10))

# Visualization
top_20 = word_counts.most_common(20)
words, counts = zip(*top_20)
plt.figure(figsize=(12, 6))
sns.barplot(x=list(counts), y=list(words), palette="viridis")
plt.title("Top 20 Most Frequent Words in Reviews")
plt.xlabel("Frequency")
plt.ylabel("Words")
plt.tight_layout()
plt.show()

# Preprocessing validation
raw_avg = df['review_text'].str.split().str.len().mean()
clean_avg = df['processed_tokens'].str.len().mean()
print(f"\nPreprocessing Validation:")
print(f"  Average tokens before cleaning: {raw_avg:.1f}")
print(f"  Average tokens after cleaning:  {clean_avg:.1f}")
print(f"  Vocabulary size after cleaning: {len(word_counts)}")

# Term frequency by rating class
rating_tokens = defaultdict(list)
for tokens, rating in zip(df['processed_tokens'], df['rating']):
    rating_tokens[rating].extend(tokens)

print("\nTop 8 words per star rating:")
for r in sorted(rating_tokens.keys()):
    top = Counter(rating_tokens[r]).most_common(8)
    print(f"{int(r)}-star: {top}")

# --- Part 2: Time Series ---
# Refresh the dataframe and ensure timestamp formatting
df = pd.read_csv('amazon_reviews_lab.csv')
df = df.dropna(subset=['timestamp', 'rating'])
# coerce errors so bad dates become NaT, then drop them
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce') 
df = df.dropna(subset=['timestamp'])

print(f"\nEarliest review: {df['timestamp'].min()}")
print(f"Latest review:  {df['timestamp'].max()}")

df = df.set_index('timestamp')

# Use 'ME' (Month End) instead of 'M' to avoid Pandas FutureWarnings
monthly_review_count = df.resample('ME').size()
monthly_avg_rating   = df['rating'].resample('ME').mean()

# Volume plot
plt.figure(figsize=(12, 5))
monthly_review_count.plot(color='blue')
plt.title('Monthly Review Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Reviews')
plt.grid(True)
plt.tight_layout()
plt.show()

# Rating trend
rolling = monthly_avg_rating.rolling(window=3).mean()
plt.figure(figsize=(12, 5))
monthly_avg_rating.plot(label='Original Monthly Avg', color='lightgray', marker='o')
rolling.plot(label='3-Month Rolling Avg', color='red', linewidth=2)
plt.title('Average Monthly Rating Over Time')
plt.xlabel('Date')
plt.ylabel('Average Rating (1-5)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
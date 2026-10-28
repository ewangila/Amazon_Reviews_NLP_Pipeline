import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter, defaultdict
from nltk.util import ngrams
import warnings

warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


# 1. Load data
df = pd.read_csv('amazon_reviews_lab.csv')

print(f"Dataset Shape: {df.shape}")
print("\nRating Value Counts:\n", df['rating'].value_counts())
print("\nFirst 3 rows:\n", df.head(3))


# 2. Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w.isalpha()]
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return tokens

df['processed_tokens'] = df['review_text'].apply(preprocess_text)


# 3. Overall frequency analysis
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


# 4. Term frequency by rating class
rating_tokens = defaultdict(list)
for tokens, rating in zip(df['processed_tokens'], df['rating']):
    rating_tokens[rating].extend(tokens)

print("\nTop 8 words per star rating:")
for r in sorted(rating_tokens.keys()):
    top = Counter(rating_tokens[r]).most_common(8)
    print(f"\n{r}-star: {top}")

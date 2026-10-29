import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
import warnings

warnings.filterwarnings('ignore')


# 1. Load data
df = pd.read_csv('amazon_reviews_lab.csv')
df = df.dropna(subset=['review_text', 'rating'])

X = df['review_text']
y = df['rating'] - 1   # 1-5 → 0-4


# 2. TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)


# 3. Class weights
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, class_weights))
print("Class weights (1-5 stars):", {k+1: round(v, 2) for k, v in class_weight_dict.items()})


# 4. Baseline – Logistic Regression
print("BASELINE: Logistic Regression (with class weights)")

lr = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print(f"Test Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print("\nClassification Report (Logistic Regression):\n")
print(classification_report(y_test, y_pred_lr, target_names=['1','2','3','4','5']))


# 5. Neural Network
print("NEURAL NETWORK (MLP) – class weights + L2 + early stopping")

X_train_dense = X_train.toarray()
X_test_dense  = X_test.toarray()

model = models.Sequential([
    layers.Dense(128, activation='relu',
                 kernel_regularizer=regularizers.l2(1e-4),
                 input_shape=(X_train_dense.shape[1],)),
    layers.Dropout(0.4),
    layers.Dense(64, activation='relu',
                 kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.4),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(5, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True, verbose=1
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5, verbose=1
)

history = model.fit(
    X_train_dense, y_train,
    epochs=25,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test_dense, y_test, verbose=0)
print(f"\nNeural Net Test Accuracy: {test_acc:.4f}")

y_pred_nn = np.argmax(model.predict(X_test_dense, verbose=0), axis=1)
print("\nClassification Report (Neural Network):\n")
print(classification_report(y_test, y_pred_nn, target_names=['1','2','3','4','5']))


# 6. Visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5])
axes[0].set_title('Logistic Regression\n(with class weights)')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

cm_nn = confusion_matrix(y_test, y_pred_nn)
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5])
axes[1].set_title('Neural Network (MLP)\n(with class weights + regularization)')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Curves')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.show()


# 7. Quantitative comparison table
print("MODEL COMPARISON SUMMARY")
print(f"{'Model':<25} {'Accuracy':>10} {'Macro-F1':>10}")
print(f"{'Logistic Regression':<25} {accuracy_score(y_test, y_pred_lr):>10.4f} "
      f"{f1_score(y_test, y_pred_lr, average='macro'):>10.4f}")
print(f"{'Neural Network (MLP)':<25} {test_acc:>10.4f} "
      f"{f1_score(y_test, y_pred_nn, average='macro'):>10.4f}")
print("="*50)

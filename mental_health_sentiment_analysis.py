#import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from summarytools import dfSummary
from wordcloud import WordCloud
from collections import Counter
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('punkt')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_auc_score

#read file
df = pd.read_csv(r"C:\Users\NANA\Downloads\Combined Data.csv")
df.head()
#summary statistics
dfSummary(df) # there are no duplicates, 362(0.7%) missing values and 7 uniques values in status column
#dropping missing values
df = df.dropna()
df.isnull().sum()

"""
# Get distribution of status
word_freq = Counter(df['status'])
print(word_freq)
distribution = word_freq.most_common()
print(   )
print(common_words)

status, counts = zip(*distribution)
plt.figure(figsize=(10, 5))
plt.bar(status, counts, color='skyblue')
plt.xlabel("Status")
plt.ylabel("Frequency")
plt.title("Mental Health Status Distribution")
plt.xticks(rotation=45)
plt.show()

"""

#Mental Health Status Distribution
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='status', order=df['status'].value_counts().index, palette='viridis')
plt.title("Mental Health Status Distribution")
plt.xlabel("Mental Health Status")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

#find frequent words in statement 
text = " ".join(statement for statement in df['statement'].astype(str))
plt.figure(figsize=(10, 6))
wordcloud = WordCloud(width=800, height=400, background_color="black", colormap="inferno").generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("Word Cloud of All Statements")
plt.axis("off")
plt.show()

#Checking for most common words in each statement and status combination
def most_common_words(text):
    stop_words = set(stopwords.words('english'))
    words = text.lower().split()
    words = [word for word in words if word not in stop_words and word not in string.punctuation ]
    word_counts = Counter(words)
    return word_counts.most_common(10)

for status in df['status'].unique():
    status_text = " ".join(df[df['status'] == status]['statement'].astype(str))
    common_words = most_common_words(status_text)
    words, counts = zip(*common_words)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(words), y=list(counts), palette='viridis')
    plt.title(f"Top 10 Common Words in {status} Statements")
    plt.xlabel("Words")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

#check length of statement based on mental health status
df['statement_length'] = df['statement'].astype(str).apply(len)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='status', y='statement_length', palette='viridis')
plt.title("Statement Length Based on Mental Health Status")
plt.xlabel("Mental Health Status")
plt.ylabel("Statement Length")
plt.xticks(rotation=45)
plt.show()

#preprocessing text
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    
    words = text.split()
    text = ' '.join([lemmatizer.lemmatize(word) for word in words if word not in stop_words])
    
    return text
df['statement'] = df['statement'].astype(str).apply(preprocess_text)
df = df.drop(['Unnamed: 0'],axis=1)
#using one hot encoding on status column
df_encoded = pd.get_dummies(df, columns=['status'], prefix='status')
#df.columns
df_encoded = df_encoded.drop(['statement_length'],axis =1)
#defining dependent and independent variables

X = df_encoded.drop(columns=[col for col in df_encoded.columns if col.startswith('status_')])

# Dependent variable (y): All the 'status_*' columns
y = df_encoded[[col for col in df_encoded.columns if col.startswith('status_')]]

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# TF-IDF Vectorizer for 'statement' text
vectorizer = TfidfVectorizer(max_features=5000,stop_words='english',ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train['statement'])
X_test_tfidf = vectorizer.transform(X_test['statement'])

#fit naive bayes
# Naive Bayes Model with OneVsRestClassifier
nb_model = OneVsRestClassifier(MultinomialNB())
nb_model.fit(X_train_tfidf, y_train)
y_pred_train = nb_model.predict(X_train_tfidf)
print("Training Accuracy:", accuracy_score(y_train, y_pred_train))
nb_pred = nb_model.predict(X_test_tfidf)

# Logistic Regression Model with OneVsRestClassifier
lr_model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
lr_model.fit(X_train_tfidf, y_train)
lr_pred = lr_model.predict(X_test_tfidf)

# SVM Model with OneVsRestClassifier
svm_model = OneVsRestClassifier(SVC(kernel='linear'))
svm_model.fit(X_train_tfidf, y_train)
svm_pred = svm_model.predict(X_test_tfidf)

# Evaluate models
print("\nNaive Bayes - Accuracy:", accuracy_score(y_test, nb_pred))
print("\nNaive Bayes- Area Under the Curve:", roc_auc_score(y_test, nb_pred))
print("\nLogistic Regression - Accuracy:", accuracy_score(y_test, lr_pred))
print("\nLogistic Regression- Area Under the Curve:",roc_auc_score(y_test, lr_pred))

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
from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_auc_score
from sklearn.preprocessing import LabelEncoder

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
df = df.drop(['Unnamed: 0','statement_length'],axis=1)
#using one hot encoding on status column
label_encoder = LabelEncoder()
df['status']= label_encoder.fit_transform(df['status'])
#df_encoded = pd.get_dummies(df, columns=['status'], prefix='status')
#df.columns
#df_encoded = df_encoded.drop(['statement_length'],axis =1)
#defining dependent and independent variables
#X = df_encoded.drop(columns=[col for col in df_encoded.columns if col.startswith('status_')])

# Dependent variable (y): All the 'status_*' columns
#y = df_encoded[[col for col in df_encoded.columns if col.startswith('status_')]]
X = df['statement']
y = df['status']
# For Naive Bayes, we will use LabelEncoder for target variable

#y_nb = label_encoder.fit_transform(df['status'])

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# TF-IDF Vectorizer for 'statement' text
vectorizer = TfidfVectorizer(max_features=5000,stop_words='english',ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Define parameter grids for GridSearchCV
from sklearn.pipeline import Pipeline

# pipeline for Naive Bayes with TfidfVectorizer
#pipeline_nb = Pipeline([
#    ('vectorizer', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
#    ('nb', MultinomialNB())
#])

#HYPERPARAMETERS

#For naive bayes
param_grid_nb = {
    'alpha': [0.5, 1.0, 1.5]  # Smoothing parameter for Naive Bayes
}

# for logistic regression 
param_grid_lr = {
    'C': [0.1, 1.0, 10.0],  
    'solver': ['liblinear', 'saga'],  
    'max_iter': [500, 1000]  
}

# For XGBoost
param_dist_xgb = {
    'learning_rate': [0.1],
    'n_estimators': [100],
    'max_depth': [5],
    'subsample': [0.8],
    'colsample_bytree': [0.8]
}

# For Random Forest
param_dist_rf = {
    'n_estimators': [50],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

"""
param_grid_svm = {
    'C': [0.1, 1.0, 10.0],           # Regularization strength
    'kernel': ['linear', 'rbf'],     # Try both linear and RBF
    'gamma': ['scale', 'auto']       # Kernel coefficient for ‘rbf’
}
""" 

# Naive Bayes with GridSearchCV
naive_model = MultinomialNB()
grid_search_nb = GridSearchCV(naive_model, param_grid_nb, cv=3, n_jobs=-1, verbose=1)
grid_search_nb.fit(X_train_tfidf, y_train)  
best_nb_model = grid_search_nb.best_estimator_
y_pred_nb = best_nb_model.predict(X_test_tfidf)

# Logistic Regression with GridSearchCV
lr_model = LogisticRegression(max_iter=1000)
grid_search_lr = GridSearchCV(lr_model, param_grid_lr, cv=3, n_jobs=-1, verbose=1)
grid_search_lr.fit(X_train_tfidf, y_train)
best_lr_model = grid_search_lr.best_estimator_
y_pred_lr = best_lr_model.predict(X_test_tfidf)

"""
# SVM with GridSearchCV
svm_model = SVC(kernel='linear')
grid_search_svm = GridSearchCV(svm_model, param_grid_svm, cv=3, n_jobs=-1, verbose=1)
grid_search_svm.fit(X_train_tfidf, y_train)
best_svm_model = grid_search_svm.best_estimator_
y_pred_svm = best_svm_model.predict(X_test_tfidf) """

# XGBoost with RandomizedSearchCV
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
random_search_xgb = RandomizedSearchCV(xgb_model, param_distributions=param_dist_xgb, n_iter=10, cv=3, n_jobs=-1, verbose=1, random_state=42)
random_search_xgb.fit(X_train_tfidf, y_train)
best_xgb_model = random_search_xgb.best_estimator_
y_pred_xgb = best_xgb_model.predict(X_test_tfidf)

# Random Forest with RandomizedSearchCV
rf_model = RandomForestClassifier()
random_search_rf = RandomizedSearchCV(rf_model, param_distributions=param_dist_rf, n_iter=10, cv=3, n_jobs=-1, verbose=1, random_state=42)
random_search_rf.fit(X_train_tfidf, y_train)
best_rf_model = random_search_rf.best_estimator_
y_pred_rf = best_rf_model.predict(X_test_tfidf)

# Evaluate models
print("\nNaive Bayes - Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nLogistic Regression - Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\nXGBoost - Accuracy:", accuracy_score(y_test, y_pred_xgb))
#print("\nLogistic Regression- Area Under the Curve:",roc_auc_score(y_test, y_pred_lr,multi_class='ovr', average='weighted'))
#print("\nSVM  - Accuracy:", accuracy_score(y_test, y_pred_lr))
#print("\nSVM- Area Under the Curve:",roc_auc_score(y_test, y_pred_lr))

#print("\nXGBoost- Area Under the Curve:",roc_auc_score(y_test, y_pred_xgb))
#print("\nRandom forest - Accuracy:", accuracy_score(y_test, y_pred_rf))
#print("\Random forest- Area Under the Curve:",roc_auc_score(y_test, y_pred_rf))

#test
import pickle

pickle.dump(best_lr_model, open("mental_health_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

loaded_model = pickle.load(open("mental_health_model.pkl", "rb"))
loaded_tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Define the labels based on the one-hot encoding columns
labels = y_train.columns.tolist()  # or manually: labels = ['status_Sad', 'status_Happy', 'status_Angry']

# Sample text for prediction
sample_text = ["got to love the US it would be cheaper for my family to bury me than to have me treated"]
sample_tfidf = loaded_tfidf.transform(sample_text)

# Make a prediction using the loaded model
prediction = loaded_model.predict(sample_tfidf)

# Get the index of the class with the highest probability
predicted_index = np.argmax(prediction, axis=1)[0]  # Get the index of the predicted class

# Map the index to the label
predicted_status = labels[predicted_index]

print("Predicted Status:", predicted_status)
# Mental-health-Sentiment-Analysis

This project uses machine learning to predict mental health status from text-based statements. The dataset contains user statements labeled with corresponding mental health statuses such as "depression", "anxiety","suicidal",etc. The script performs data preprocessing, including text cleaning, tokenization, and lemmatization, to prepare the text data for analysis. Exploratory Data Analysis (EDA) is conducted to visualize the distribution of statuses and generate word clouds of frequent terms. Multiple machine learning models are applied, including Naive Bayes, Logistic Regression, Random Forest and XGBoost with OneVsRestClassifier for multi-class classification.

The dataset is split into training and testing sets, and model performance is evaluated using accuracy and AUC (Area Under the Curve). Visualization techniques such as bar charts and boxplots provide insights into the data distribution and statement length differences across statuses. The TfidfVectorizer is used to extract features from text data, allowing the models to better understand and classify statements. The code is written in Python and relies on libraries like pandas, seaborn, sklearn, and nltk for analysis and machine learning.

 The final output includes model evaluation metrics to interpret the results

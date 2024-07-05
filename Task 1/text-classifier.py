import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils import resample
import re

# Function to perform data augmentation for categories with fewer samples
def augment_data(category, df):
    category_data = df[df['category'] == category]
    num_samples = len(category_data)
    if num_samples < 5000:
        # Upsample the category to have at least 5000 samples
        category_augmented = resample(category_data, n_samples=5000-num_samples, random_state=42)
        df = pd.concat([df, category_augmented])
    return df

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Tokenize the text (split by whitespace)
    tokens = re.findall(r'\b\w+\b', text)
    return ' '.join(tokens)

# Load the data from the news folder
parent_folder = r'C:\Users\seyma\IdeaProjects\havelsan-hackathon\Task 1\sınıflandırma_train_data'
data = []

for category in os.listdir(os.path.join(parent_folder, 'news')):
    category_path = os.path.join(parent_folder, 'news', category)
    for file_name in os.listdir(category_path):
        file_path = os.path.join(category_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        data.append((text, category))

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=['text', 'category'])
df_input = pd.read_csv('siniflandirma.csv', sep='|')

# Perform data augmentation for categories with fewer samples
categories_to_augment = ['dunya', 'ekonomi', 'genel', 'guncel', 'kultur-sanat', 'magazin', 'planet', 'saglik', 'spor', 'teknoloji', 'turkiye', 'yasam']  # Adjust as needed
for category in categories_to_augment:
    df = augment_data(category, df)

# Preprocess the text data
df['text_processed'] = df['text'].apply(preprocess_text)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text_processed'], df['category'], test_size=0.2, random_state=42)

# Convert text data into numerical features using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a logistic regression classifier
param_grid = {'C': [100]} # {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(LogisticRegression(solver='lbfgs', max_iter=1000), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_tfidf, y_train)
best_logistic_classifier = grid_search.best_estimator_
print(grid_search.best_params_)

# Evaluate the best logistic regression classifier
y_pred_logistic = best_logistic_classifier.predict(X_test_tfidf)
print("Classification Report for Logistic Regression:")
print(classification_report(y_test, y_pred_logistic))

# Load and preprocess the input CSV file
input_csv_path = 'siniflandirma.csv'
input_df = pd.read_csv(input_csv_path, sep='|')
input_df['metin'] = input_df['metin'].apply(preprocess_text)

# Convert text data into TF-IDF features
X_input_tfidf = tfidf_vectorizer.transform(input_df['metin'])

# Predict categories for the input data
predicted_categories = best_logistic_classifier.predict(X_input_tfidf)
input_df['kategori'] = predicted_categories

# Save the results to a new CSV file
output_csv_path = 'sonuclar.csv'
input_df.to_csv(output_csv_path, index=False)

print("Predicted categories for the input CSV file saved to:", output_csv_path)


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to perform semantic search
def semantic_search(data, query_column='text', top_n=5):
    # Preprocess text data (simple preprocessing: lowercasing)
    data['processed_text'] = data[query_column].str.lower()

    # Create document embeddings using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['processed_text'])

    def search(query):
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, X)
        indices = similarities.argsort()[0][-top_n:][::-1]
        return data.iloc[indices]['title'].tolist()
    return search

# Load data from input CSV file
input_file_path = "anlamsal_arama_data.csv"
output_file_path = "sonuclar.csv"
data = pd.read_csv(input_file_path, sep='|')

# Perform semantic search
search_function = semantic_search(data)

# Read queries from CSV file
query_file_path = "task2_yarismaci_test_data.csv"
queries_df = pd.read_csv(query_file_path, sep='|')

# Perform semantic search for each query
results = []
for index, row in queries_df.iterrows():
    query = row['sorgu']
    result_titles = search_function(query)
    results.append({'sorgu': query, '1st': result_titles[0], '2nd': result_titles[1], '3rd': result_titles[2],
                    '4th': result_titles[3], '5th': result_titles[4]})

# Write results back to the same CSV file
results_df = pd.DataFrame(results)
results_df.to_csv(output_file_path, index=False, sep='|')

print(f"Semantic search results have been written to {output_file_path}")
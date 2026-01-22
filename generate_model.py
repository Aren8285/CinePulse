import pandas as pd
import pickle
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_model():
    print("Step 1/5: Loading datasets...")
    try:
        movies = pd.read_csv('data/tmdb_5000_movies.csv')
        credits = pd.read_csv('data/tmdb_5000_credits.csv')
    except FileNotFoundError:
        print("Error: Files not found in 'data/' folder. Please download them from Kaggle.")
        return

    # Merge datasets
    movies = movies.merge(credits, on='title')
    
    # Select critical columns
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)

    # --- HELPER FUNCTIONS ---
    def convert(obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L

    def convert3(obj):
        L = []
        counter = 0
        for i in ast.literal_eval(obj):
            if counter != 3:
                L.append(i['name'])
                counter += 1
            else:
                break
        return L
    
    def fetch_director(obj):
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
        return L

    print("Step 2/5: Processing tags (this might take a moment)...")
    # Extract details from JSON format
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    # Collapse spaces to create unique tags (e.g., "Science Fiction" -> "ScienceFiction")
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

    # Create the Master Tag
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    
    # Create clean dataframe
    new_df = movies[['movie_id', 'title', 'tags']]
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

    print("Step 3/5: Vectorizing data...")
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()

    print("Step 4/5: Calculating similarity matrix...")
    similarity = cosine_similarity(vectors)

    print("Step 5/5: Saving model files...")
    pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))
    pickle.dump(similarity, open('similarity.pkl', 'wb'))
    
    print("✅ Success! 'movie_dict.pkl' and 'similarity.pkl' created.")

if __name__ == "__main__":
    create_model()

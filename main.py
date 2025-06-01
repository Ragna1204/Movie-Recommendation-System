import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Load datasets
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

# Merge datasets
movies = movies.merge(credits, on='title')
print("Merged Dataframe:", movies.shape)

# Select only important columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

# Safe parsing functions
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def get_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return i['name']
    return ''

# Parse the JSON-like columns
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: convert(x)[:3])  # Top 3 actors
movies['crew'] = movies['crew'].apply(get_director)

# Ensure overview is string
movies['overview'] = movies['overview'].astype(str)

# Create 'tags' column by combining all important text info
movies['tags'] = movies['overview'] + ' ' + \
                 movies['genres'].apply(lambda x: ' '.join(x)) + ' ' + \
                 movies['keywords'].apply(lambda x: ' '.join(x)) + ' ' + \
                 movies['cast'].apply(lambda x: ' '.join(x)) + ' ' + \
                 movies['crew']

# Show result
print(movies[['title', 'tags']].head())

# Convert tags to lowercase and remove spaces in multi-word terms
movies['tags'] = movies['tags'].apply(lambda x: x.lower())

# Optional: Remove spaces in tags to treat multi-word names as one
def remove_spaces(L):
    return [i.replace(" ", "") for i in L]

movies['genres'] = movies['genres'].apply(remove_spaces)
movies['keywords'] = movies['keywords'].apply(remove_spaces)
movies['cast'] = movies['cast'].apply(remove_spaces)
movies['crew'] = movies['crew'].apply(lambda x: x.replace(" ", ""))

# Recreate the cleaned 'tags' column
movies['tags'] = movies['overview'] + ' ' + \
                 movies['genres'].apply(lambda x: ' '.join(x)) + ' ' + \
                 movies['keywords'].apply(lambda x: ' '.join(x)) + ' ' + \
                 movies['cast'].apply(lambda x: ' '.join(x)) + ' ' + \
                 movies['crew']

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)

# Function to find similar movies
def recommend(movie):
    movie = movie.lower()
    if movie not in movies['title'].str.lower().values:
        print("Movie not found in database.")
        return

    index = movies[movies['title'].str.lower() == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    print(f"\nTop 5 recommendations for '{movies.iloc[index].title}':")
    for i in movie_list:
        print(movies.iloc[i[0]].title)


recommend("notting hill")
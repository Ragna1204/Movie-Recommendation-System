import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess data (reuse your code here)
@st.cache_data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)
    def convert(obj):
        return [i['name'] for i in ast.literal_eval(obj)]

    def get_director(obj):
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return i['name']
        return ''

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(lambda x: convert(x)[:3])
    movies['crew'] = movies['crew'].apply(get_director)
    movies['overview'] = movies['overview'].astype(str)

    def remove_spaces(L):
        return [i.replace(" ", "") for i in L]

    movies['genres'] = movies['genres'].apply(remove_spaces)
    movies['keywords'] = movies['keywords'].apply(remove_spaces)
    movies['cast'] = movies['cast'].apply(remove_spaces)
    movies['crew'] = movies['crew'].apply(lambda x: x.replace(" ", ""))

    movies['tags'] = movies['overview'] + ' ' + \
                     movies['genres'].apply(lambda x: ' '.join(x)) + ' ' + \
                     movies['keywords'].apply(lambda x: ' '.join(x)) + ' ' + \
                     movies['cast'].apply(lambda x: ' '.join(x)) + ' ' + \
                     movies['crew']

    movies['tags'] = movies['tags'].apply(lambda x: x.lower())

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()

    similarity = cosine_similarity(vectors)

    return movies, similarity

movies, similarity = load_data()

def recommend(movie):
    movie = movie.lower()
    if movie not in movies['title'].str.lower().values:
        return []
    index = movies[movies['title'].str.lower() == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended = [movies.iloc[i[0]].title for i in movie_list]
    return recommended


# Streamlit UI
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Select or type a movie you like:",
    movies['title'].values
)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    if recommendations:
        st.write(f"Top 5 movies similar to **{selected_movie}**:")
        for i, movie in enumerate(recommendations, start=1):
            st.write(f"{i}. {movie}")
    else:
        st.write("Movie not found in database, try another one.")
import pandas as pd
import numpy as np

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

credits.head()
movies.head()

print("Credits:",credits.shape)
print("Movies Dataframe:",movies.shape)

movies = movies.merge(credits, left_on='title', right_on='title')

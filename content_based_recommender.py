# %% [markdown]
# 1. Preprocessing for Model Building
# 

# %%
# Convert Genres to Vector Format:

# Use one-hot encoding or count vectorizer to transform genres into a numerical format.
# Example: Adventure|Animation → [1, 1, 0, 0, ...]


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer


final_data = pd.read_csv('datasets/final_data.csv')
final_data['genres_list'] = final_data['genres'].apply(lambda x: x.split('|'))
mlb = MultiLabelBinarizer()
genres_encoded = pd.DataFrame(mlb.fit_transform(final_data['genres_list']), columns=mlb.classes_)
final_data = pd.concat([final_data, genres_encoded], axis=1)

# %% [markdown]
# pd.to_datetime(): Converts the rating_timestamp column from string to datetime64 format.
# 
# .astype(int): Converts the datetime objects to integers representing the number of nanoseconds since the epoch. Dividing by 10^9 converts nanoseconds to seconds.
# 
# MinMaxScaler: Scales both rating and the converted rating_timestamp to a range between 0 and 1.

# %%
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Convert the 'rating_timestamp' column to numeric (seconds since epoch)
final_data['rating_timestamp'] = pd.to_datetime(final_data['rating_timestamp']).astype('int64') // 10**9

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Apply MinMaxScaler to 'rating' and 'rating_timestamp'
final_data[['rating', 'rating_timestamp']] = scaler.fit_transform(final_data[['rating', 'rating_timestamp']])

# Check the scaled values
print(final_data[['rating', 'rating_timestamp']].head())


# %%
# Prepare User-Item Matrix:

# Create a matrix where rows are userId, columns are movieId, and values are ratings.
# This matrix is the input for collaborative filtering.

user_item_matrix = final_data.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)
print(user_item_matrix)

# %% [markdown]
# 2. Splitting the Data

# %%
# Split your data into training and testing sets:

# For user-item matrices, consider splitting on user interactions or time.
from surprise.model_selection import train_test_split

reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(final_data[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

# %% [markdown]
# 3. Choose a Recommendation Algorithm
# 
# 
#     Decide which approach fits your goals:
# 
#     Collaborative Filtering:
#         Uses user-item interaction data.
# 
#     Techniques:
#         User-Based Collaborative Filtering: Recommends based on similar users.
# 
#         Item-Based Collaborative Filtering: Recommends based on similar items.
# 

# %% [markdown]
# 1. Collaborative Filtering
# Collaborative filtering relies on user-item interaction data, such as movie ratings, to recommend items. It assumes that if users share similar preferences for some items, they may like other similar items.
# 
# Techniques:
# User-Based Collaborative Filtering: Recommends movies that users with similar tastes liked.
# 
# Example: If User A and User B both like movies X and Y, and User A also likes Z, then Z will be recommended to User B.
# Steps:
# Calculate similarity between users (e.g., cosine similarity, Pearson correlation).
# Identify the nearest neighbors (users most similar to the target user).
# Aggregate their preferences to make recommendations.
# Item-Based Collaborative Filtering: Recommends movies similar to those the user has already rated highly.
# 
# Example: If movies X and Y are rated similarly by many users, and User A likes X, then Y will be recommended to User A.
# Steps:
# Calculate similarity between items (e.g., cosine similarity, Pearson correlation).
# Find similar items to those the target user liked.
# Aggregate ratings for those items to recommend.
# Implementation Example:
# Using the Surprise library for collaborative filtering with the SVD algorithm:

# %% [markdown]
# SVD (Singular Value Decomposition): A matrix factorization technique that reduces the dimensionality of the user-item interaction matrix to discover latent features.

# %%
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# Example dataset
data_dict = {
    'userId': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    'movieId': [101, 102, 103, 101, 104, 102, 103, 105, 101, 105],
    'rating': [4.0, 5.0, 3.0, 4.0, 5.0, 2.0, 4.0, 5.0, 3.0, 4.0]
}
ratings_df = pd.DataFrame(data_dict)

# Convert data into Surprise Dataset format
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2)

# Train SVD model
model = SVD()
model.fit(trainset)

# Predict ratings for the testset
predictions = model.test(testset)

# Create a recommendation function
def recommend_movies_collaborative(user_id, model, data, n_recommendations=5):
    # Get unique movieIds
    all_movie_ids = set(data['movieId'])
    
    # Get movies rated by the user
    rated_movie_ids = set(data[data['userId'] == user_id]['movieId'])
    
    # Find movies not rated by the user
    unrated_movie_ids = list(all_movie_ids - rated_movie_ids)
    
    # Predict ratings for unrated movies
    predicted_ratings = [
        (movie_id, model.predict(user_id, movie_id).est)
        for movie_id in unrated_movie_ids
    ]
    
    # Sort by predicted rating
    predicted_ratings = sorted(predicted_ratings, key=lambda x: x[1], reverse=True)
    
    # Return top N recommendations
    top_recommendations = [movie_id for movie_id, rating in predicted_ratings[:n_recommendations]]
    return top_recommendations

# Recommend movies for a specific user
user_id = 1  # Change this to the userId you want to test
recommendations = recommend_movies_collaborative(user_id, model, ratings_df)
print(f"Recommended movies for user {user_id}: {recommendations}")


# %% [markdown]
# 2. Content-Based Filtering
# Content-based filtering uses the features of items (e.g., genres, tags) to recommend similar items.
# 
# Steps:
# Extract features from the data (e.g., genres, tags).
# Create a profile for each user based on their rated items.
# Recommend items similar to the user’s preferences using a similarity metric (e.g., cosine similarity, Euclidean distance)
# 

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create TF-IDF matrix for genres
tfidf = TfidfVectorizer(tokenizer=lambda x: x.split('|'))
tfidf_matrix = tfidf.fit_transform(final_data['genres'])

# Calculate similarity between movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommend movies for a given movie index
def recommend_movies(movie_idx, cosine_sim, data, top_n=10):
    similar_movies = list(enumerate(cosine_sim[movie_idx]))
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    recommendations = [data.iloc[i[0]].title for i in similar_movies[1:top_n+1]]
    return recommendations

movie_idx = 0  # Index of a specific movie
recommendations = recommend_movies(movie_idx, cosine_sim, final_data)
print(recommendations)


# %% [markdown]
# 3. Hybrid Filtering
# Hybrid filtering combines collaborative and content-based filtering to address their limitations. It leverages both user-item interaction data and item features.
# 
# Approaches:
# Combine predictions from collaborative and content-based models.
# Use content-based filtering to preprocess data for collaborative filtering.
# Build a single model that integrates both approaches.

# %%
# Assuming specific_user_id is the user ID for whom we are making predictions
specific_user_id = 1  # Replace with the actual user ID

# Get indices of movies rated by this user
user_movie_indices = final_data[final_data['userId'] == specific_user_id].index.tolist()

# Content-based predictions using cosine similarity
content_pred = []
for user_rated_idx in user_movie_indices:
    # Ensure movie index is within bounds of cosine_sim
    if user_rated_idx < cosine_sim.shape[0]:
        # Calculate the mean similarity score for each movie
        movie_similarities = cosine_sim[user_rated_idx]
        content_pred.append(movie_similarities.mean())
    else:
        content_pred.append(0)  # Fallback for out-of-bounds indices

# Collaborative filtering predictions for the specific user
collaborative_pred = [
    pred.est for pred in model.test(testset) if pred.uid == specific_user_id
]

# Ensure predictions are normalized to the same length
if len(collaborative_pred) != len(content_pred):
    print(f"Mismatched lengths: Collaborative={len(collaborative_pred)}, Content={len(content_pred)}")
    min_len = min(len(collaborative_pred), len(content_pred))
    collaborative_pred = collaborative_pred[:min_len]
    content_pred = content_pred[:min_len]

# Final hybrid prediction
final_pred = [0.5 * c + 0.5 * t for c, t in zip(collaborative_pred, content_pred)]

print("Final hybrid predictions:", final_pred)


# %%
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example dataset of movies with their genres
data_dict = {
    'movieId': [101, 102, 103, 104, 105],
    'title': ["Movie 1", "Movie 2", "Movie 3", "Movie 4", "Movie 5"],
    'genres': ["Action|Adventure", "Action|Thriller", "Drama|Romance", "Comedy|Drama", "Thriller|Action"]
}
final_data = pd.DataFrame(data_dict)

# TF-IDF Vectorization for genres
tfidf = TfidfVectorizer(tokenizer=lambda x: x.split('|'))
tfidf_matrix = tfidf.fit_transform(final_data['genres'])

# Calculate cosine similarity between movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommend movies based on cosine similarity
def recommend_movies_content_based(movie_idx, cosine_sim, data, top_n=5):
    similar_movies = list(enumerate(cosine_sim[movie_idx]))
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    recommendations = [data.iloc[i[0]].title for i in similar_movies[1:top_n+1]]
    return recommendations

# Streamlit interface
st.title("Content-Based Movie Recommendation System")

# Movie selection from the sidebar
st.sidebar.header("Select a Movie")
movie_titles = final_data['title'].tolist()
selected_movie = st.sidebar.selectbox("Choose a Movie", movie_titles)

# Get the index of the selected movie
selected_movie_idx = final_data[final_data['title'] == selected_movie].index[0]

# When button is pressed, get recommendations
if st.sidebar.button("Get Recommendations"):
    recommendations = recommend_movies_content_based(selected_movie_idx, cosine_sim, final_data)
    st.write(f"Movies similar to '{selected_movie}':")
    for i, movie in enumerate(recommendations, 1):
        st.write(f"{i}. {movie}")





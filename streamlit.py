import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

movies = pd.read_csv('movies.csv')
required_columns =['title','original_title', 'tagline', 'keywords', 'overview', 'genres', 'cast', 'director']
movies = movies[required_columns]
movies.fillna(' ', inplace=True)
movies.isna().sum()
# movies.iloc[0]
movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['director'] = movies['director'].apply(lambda x: x.replace(" ","") )


movies['genres'] = movies['genres'].apply(lambda x: ','.join(map(str, x)))
movies['keywords'] = movies['keywords'].apply(lambda x: ','.join(map(str, x)))
movies['cast'] = movies['cast'].apply(lambda x: ','.join(map(str, x)))

movies['content'] = movies['title'] + ' ' + movies['overview'] + ' ' + movies['keywords'] + ' ' + movies['cast'] + ' ' + movies ['director'] 

vectorizer = TfidfVectorizer(max_features=1000)
movie_vectors = vectorizer.fit_transform(movies['content'].values) 


similarity = cosine_similarity(movie_vectors)

similarity_df = pd.DataFrame(similarity)
def recommend(movie):
    # find movie index from dataset
    movies_index = movies[movies['title'] == movie].index[0]
    
    # finding cosine similarities of movie
    distances = similarity[movies_index]
    
    # sorting cosine similarities
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:15]
    
    for i in movies_list:
       st.write(movies.iloc[i[0]].title)
st.write("Netflix Recommendation System")
user_input = st.text_input("Enter Movie")
if user_input:
    try:
        recommend(user_input)
    except:
        st.write("Enter Another Movie Name")


from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import pickle 
from sklearn.externals import joblib


def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]


#define a sample text
text = ["London Paris London ", "Paris Paris London"]

#cv is the object of class called countvectorizer
cv= CountVectorizer()

#a variable to store the vector of text
count_matrix = cv.fit_transform(text)

#to array will convert matrix into an array
print ((count_matrix).toarray())

similarity_scores = cosine_similarity(count_matrix)

#print(similarity_scores)




#read dataset
df = pd.read_csv(r"C:/Users/Ankur/Desktop/ML/movie_dataset.csv")
df = df.iloc[ : , 0:25]
print (df.columns)

#Select some features
features = ['keywords', 'cast', 'genres', 'director']

for feature in features:
	df[feature] = df[feature].fillna(' ')

#Create a column in ddataframe to combine features
def combine_features(row):
	return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " "+row['director'] 


df["combine_features"]= df.apply(combine_features, axis = 1)



print(df["combine_features"].head())


cv= CountVectorizer()

count_matrix = cv.fit_transform(df["combine_features"])


cosine_sim = cosine_similarity(count_matrix)


print((count_matrix).toarray())

movie_user_likes = "Pirates of the Caribbean: At World's End"

#get index of movie

movie_index = get_index_from_title(movie_user_likes)


print(movie_index)


#list of tuples of similar movies

similar_movies = list(enumerate(cosine_sim[1]))


#sort the tuple
sorted_similar_movies = sorted(similar_movies,key= lambda x:x[1], reverse=True)

with open('C:/Users/Ankur/Desktop/ML/mrmodel.pkl', 'wb') as f:
    pickle.dump(sorted_similar_movies, f)



print(sorted_similar_movies)
#print the titles
i=0
for movie in sorted_similar_movies:
	print(get_title_from_index(movie[0]))
	i = i +1
	if i>5:
		break
    



    

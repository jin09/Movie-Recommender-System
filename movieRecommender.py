import pandas as pd

# get the user data 
r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('/ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

m_cols = ['movie_id', 'title']
movies = pd.read_csv('/ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

ratings = pd.merge(movies, ratings)

# print a fraction of data got
ratings.head() 

# real magic of pandas where I am going to create a new data frame
# where rows will have all the users and columns will have all the movies 
# internal cells will have all the ratings that user gave to a particular movie

movieRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
movieRatings.head()

# If I correlate rows I get similar users,
# If I correlate columns I get similar movies, 
# so forst we extract the movie that we want to check for

starWarsRatings = movieRatings['Star Wars (1977)']
starWarsRatings.head()

# now find correlation between the movie star wars and all others

similarMovies = movieRatings.corrwith(starWarsRatings)
similarMovies = similarMovies.dropna()
df = pd.DataFrame(similarMovies)
df.head(10)

# lets sort the values by correlation so that we have answer on top

similarMovies.sort_values(ascending=False)

# Wait, there are so many with perfect correlation, What should I do?

# when I look at the result, I can say that even if 2 people saw 2 movies and rated them perfectly
# even then I am calling those 2 movies similar

# better way would be to consider movies that have received a good amount of reviews.
# maybe, consider movies only that were rated/seen by over 100 people(based on metrics)

import numpy as np
movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})
movieStats.head()

# filter movies < 100 reviews

popularMovies = movieStats['rating']['size'] >= 100
movieStats[popularMovies].sort_values([('rating', 'mean')], ascending=False)[:15]

# we make a new data frame with similarity where star wars will be correlated to ratings of more than 100 users

df = movieStats[popularMovies].join(pd.DataFrame(similarMovies, columns=['similarity']))
df.head()

# sort by similarity

df.sort_values(['similarity'], ascending=False)[:15]



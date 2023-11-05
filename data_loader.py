import pandas as pd
import numpy as np
import os
import json

## Define the paths to the datasets

CMU_PATH = 'datasets/MovieSummaries/'
CHARACTER_DATASET = CMU_PATH+"character.metadata.tsv"
MOVIE_DATASET = CMU_PATH+"movie.metadata.tsv"
NAME_CLUSTER_DATASET = CMU_PATH+"name.clusters.txt"
PLOT_SUMMARY_DATASET = CMU_PATH+"plot_summaries.txt"
TROPES_CLUSTER_DATASET = CMU_PATH+"tvtropes.clusters.txt"
#------
#We are not using this dataset finally
IMDB_PATH = 'datasets/imdb/'
IMDB_MOVIE = IMDB_PATH+"movies.csv"
IMDB_RATING = IMDB_PATH+"ratings.csv"
#------

KAGGLE_MOVIE_PATH = 'datasets/kaggle_movie/'
KAGGLE_MOVIE = KAGGLE_MOVIE_PATH+"movies_metadata.csv"
KAGGLE_RATING = KAGGLE_MOVIE_PATH+"ratings.csv"
#-----

KAGGLE_IMDB_PATH = 'datasets/kaggle_imdb/'
KAGGLE_TITLE_IMDB = KAGGLE_IMDB_PATH+"/title.basics.tsv/data.tsv"
KAGGLE_RATING_IMDB = KAGGLE_IMDB_PATH+"/title.ratings.tsv/data.tsv"
#-----

OSCAR_PATH = 'datasets/oscars_awards/'
OSCAR_WINNER = OSCAR_PATH+"the_oscar_award.csv"





## Load the datasets

def load_character():
    df_character = pd.read_csv(CHARACTER_DATASET, sep='\t', header=None)
    column_names_character = ['Wikipedia movie ID', 'Freebase movie ID', 'Release date', 'Character name', 'Birth date', 'Gender', \
                          'height [m]', 'Ethnicity (Freebase ID)', 'Actor name', 'Age at movie release', \
                          'Freebase character/actor map ID', 'Freebase character ID', 'Freebase actor ID']
    df_character.columns = column_names_character

    return df_character

def load_movie():
    df_movie = pd.read_csv(MOVIE_DATASET, sep='\t', header=None)
    column_names_movie = ['Wikipedia ID', 'Freebase ID', 'Name', 'Release date', 'Box office', 'Runtime', \
                 'Languages (Freebase ID:name tuples)', 'Countries (Freebase ID:name tuples)', \
                 'genres (Freebase ID:name tuples)']
    df_movie.columns = column_names_movie

    #convert the 'Release date' column of movie dataset to YYYY-MM-DD format with 3 new columns : 'Year', 'Day'
    df_movie['Year'] = np.nan
    df_movie['Month'] = np.nan
    df_movie['Day'] = np.nan

    #extract Year, Month, and Day based on the string format
    for index, release_date in enumerate(df_movie['Release date']):
        #if date_str is NaN, Year, Month, and Day remain NaN
        if pd.notnull(release_date):
           date_parts = str(release_date).split('-')
           df_movie.at[index, 'Year'] = int(date_parts[0])
           #if only the year is available, Month and Day are left as NaN
           if len(date_parts) == 3:  # Full date is present
              df_movie.at[index, 'Month'] = int(date_parts[1])
              df_movie.at[index, 'Day'] = int(date_parts[2])
            
    #convert columns to nullable integer types
    df_movie['Year'] = df_movie['Year'].astype('Int64')
    df_movie['Month'] = df_movie['Month'].astype('Int64')
    df_movie['Day'] = df_movie['Day'].astype('Int64')

    return df_movie

def load_name_cluster():
    df_name_cluster = pd.read_csv(NAME_CLUSTER_DATASET, delimiter=r'\s+/', header=None, engine='python')
    column_name_cluster = ['Actor name', 'Freebase character/actor map ID']
    df_name_cluster.columns = column_name_cluster

    return df_name_cluster

def load_plot_summary():
    with open(PLOT_SUMMARY_DATASET, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    column1 = []
    column2 = []


    for line in lines:
        #split on the first space
        parts = line.split(' ', 1)  
        if len(parts) == 2:
            part1 = parts[0].strip()
            part2 = parts[1].strip()
            #check if the first part is a number
            if part1.isdigit():  
                column1.append(part1)
                column2.append(part2)

    df_summary = pd.DataFrame({'Wikipedia ID movie': column1, 'Summary': column2})

    return df_summary


def load_tropes_cluster():
    with open(TROPES_CLUSTER_DATASET, 'r') as file:
      lines = file.readlines()
    
    #create empty lists to store extracted data
    column1 = []
    column2 = []
    column3 = []
    column4 = []
    column5 = []

    for line in lines:
       parts = line.strip().split('\t')
       if len(parts) == 2:
        #extract the first part
        column1.append(parts[0])
        #parse the JSON string in the second part
        data = json.loads(parts[1])
        # Extract the different values 
        column2.append(data.get('char', ''))  
        column3.append(data.get('movie', ''))  
        column4.append(data.get('id', '')) 
        column5.append(data.get('actor', '')) 
        
    df_tropes_cluster = pd.DataFrame({'Character types': column1, 'Char': column2, 'Movie': column3, 'Freebase character/actor map ID': column4, 'Actor': column5})

    return df_tropes_cluster


def load_movie_imdb():
    df_movie_id = pd.read_csv(IMDB_MOVIE, sep=',', header=0)

    df_movie_id['title'] = df_movie_id['title'].str.replace(r'\s\(\d{4}\)', '', regex=True)
    df_movie_id.rename(columns={'title': 'Name'}, inplace=True)

    return df_movie_id

def load_rating_imdb():
    df_rating = pd.read_csv(IMDB_RATING, sep=',', header=0)

    return df_rating


def load_movie_kaggle():
    df_kaggle_movie = pd.read_csv(KAGGLE_MOVIE, sep=',', header=0, low_memory=False)
    df_kaggle_movie.rename(columns={'title': 'Name', 'revenue': 'Box office', 'id': 'movieId', 'release_date': 'Release date', 'imdb_id': 'tconst'}, inplace=True)
    df_kaggle_movie['Box office'] = df_kaggle_movie['Box office'].replace(0, np.nan)
    
    #df_kaggle_movie['movieId'] = pd.to_numeric(df_kaggle_movie['movieId'], errors='coerce')
    #df_kaggle_movie.dropna(subset=['movieId'], inplace=True)
    #df_kaggle_movie['movieId'] = df_kaggle_movie['movieId'].astype(int)

    #convert the 'Release date' column of movie dataset to YYYY-MM-DD format with 3 new columns : 'Year', 'Day'
    df_kaggle_movie['Year'] = np.nan
    df_kaggle_movie['Month'] = np.nan
    df_kaggle_movie['Day'] = np.nan

    #extract Year, Month, and Day based on the string format
    for index, release_date in enumerate(df_kaggle_movie['Release date']):
        #if date_str is NaN, Year, Month, and Day remain NaN
        if pd.notnull(release_date):
           date_parts = str(release_date).split('-')
           df_kaggle_movie.at[index, 'Year'] = int(date_parts[0])
           #if only the year is available, Month and Day are left as NaN
           if len(date_parts) == 3:  # Full date is present
              df_kaggle_movie.at[index, 'Month'] = int(date_parts[1])
              df_kaggle_movie.at[index, 'Day'] = int(date_parts[2])
            
    #convert columns to nullable integer types
    df_kaggle_movie['Year'] = df_kaggle_movie['Year'].astype('Int64')
    df_kaggle_movie['Month'] = df_kaggle_movie['Month'].astype('Int64')
    df_kaggle_movie['Day'] = df_kaggle_movie['Day'].astype('Int64')
    
    return df_kaggle_movie
    
    
def load_rating_kaggle():
    df_kaggle_rating = pd.read_csv(KAGGLE_RATING, sep=',', header=0)
    
    return df_kaggle_rating
    
def load_movie_imdb_kaggle():
    df_movie_imdb = pd.read_csv(KAGGLE_TITLE_IMDB, sep='\t', header=0, low_memory=False)
    df_movie_imdb.rename(columns={'primaryTitle': 'Name'}, inplace=True)
    df_movie_imdb = df_movie_imdb[df_movie_imdb['titleType'] == 'movie']

    df_movie_imdb.rename(columns={'startYear': 'Year'}, inplace=True)
    df_movie_imdb['Year'] = pd.to_numeric(df_movie_imdb['Year'], errors='coerce').astype('Int64')


    df_movie_imdb.rename(columns={'runtimeMinutes': 'Runtime'}, inplace=True)
    df_movie_imdb['Runtime'] = pd.to_numeric(df_movie_imdb['Runtime'], errors='coerce').astype('float')

    return df_movie_imdb

def load_rating_imdb_kaggle():
    df_rating_imdb = pd.read_csv(KAGGLE_RATING_IMDB, sep='\t', header=0)

    return df_rating_imdb

def load_oscar_winner():
    df_oscar_winner = pd.read_csv(OSCAR_WINNER, sep=',', header=0, low_memory=False)
    df_oscar_winner.rename(columns={'film': 'Name', 'year_film': 'Year'}, inplace=True)
    df_oscar_winner['Year'] = pd.to_numeric(df_oscar_winner['Year'], errors='coerce').astype('Int64')
    df_oscar_winner['Year'] = df_oscar_winner['Year'].astype('Int64')
    df_oscar_winner.dropna(subset=['Name'], inplace=True)

    return df_oscar_winner
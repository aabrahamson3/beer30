# This file contains all custom functions and other imports used by my model

import pandas as pd
import numpy as np 
import seaborn as sns
from collections import defaultdict
from surprise.model_selection import KFold, cross_validate, train_test_split, GridSearchCV
from surprise import SVD, Dataset, accuracy, BaselineOnly, Reader, KNNWithMeans, KNNBasic, NormalPredictor
import re
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec, Phrases
from gensim.parsing.preprocessing import STOPWORDS as stop_words
from gensim.utils import simple_preprocess
from sklearn.feature_extraction import text
from nltk.stem.lancaster import LancasterStemmer
from gensim.models.callbacks import CallbackAny2Vec

def read_pickle_create_lookup(filepath):
    """this takes in the joined_text_df pickle file to create a lookup dict for the model"""
    df_joined = pd.read_pickle(filepath)
    lookup_df = df_joined[['id', 'brewery_id', 'name', 'city', 'state', 'country', 'brewery_name']]
    lookup_df['id'] = lookup_df['id'].astype(str)
    lookup_dict = lookup_df.set_index('id').to_dict(orient='index')
    return lookup_dict

def tag_docs(docs):
    """this takes a df with all reviews and 'text' and 'id' features as arguments makes words 
    applies the preprocesser function and then adds the beer id as tag, 
    returns a list of TaggedDocument objects"""
    results = docs.apply(lambda r: TaggedDocument(words=preprocessor(r['text']), tags=[r['id']]), axis=1)
    return results.tolist()


def stem_tag_docs(docs, my_stop_words):
    ls = LancasterStemmer()
    results = docs.apply(lambda r: TaggedDocument(words=preprocessor_and_stem(r['text'], my_stop_words), tags=[str(r['id'])]), axis=1)
    return results.tolist()

def preprocessor(text):
    """uses gensim simple_preprocess and then removes stop words
    -> used in the tag_docs function
    """
    # uses gensim simple_preprocess to lowercase and tokenize words, and then removes custom stop words
    simple = simple_preprocess(text)
    result = [word for word in simple if not word in my_stop_words]
    return result

def make_stop_words():
    global stop_words
    letters = list('abcdefghijklmnopqrstuvwxyz')
    numbers = list('0123456789')
    words = ['oz', 'ml', 'pour', 'poured', 'bottle', 'can', 'ounce',\
         'bomber', 'botttle', 'stubby', 'ouncer', 'pouring', 'growler', 'snifter',\
         'tulip', 'bottled', 'brewery', 'pint', 'glass', 'cap', 'cork']
    stopwords = stop_words.union(set(letters)).union(set(numbers)).union(set(words))
    
    my_stop_words = text.ENGLISH_STOP_WORDS.union(stopwords)
    return my_stop_words

def preprocessor_and_stem(text, my_stop_words):
    """uses gensim simple_preprocess and then removes stop words
    -> used in the tag_docs function
    """
    # Instantiate a LancasterStemmer object, use gensim simple_preprocess to tokenize/lowercase
    # and then removes stop words
    ls = LancasterStemmer()
    simple = simple_preprocess(text)
    result = [ls.stem(word) for word in simple if not word in my_stop_words]
    return result

class EpochLogger(CallbackAny2Vec):
    """ used as a callback in the doc2vec training process, just to see its progress"""
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

def output_brewery(brewery):
    """This takes a dictionary as an argument: key = brewery_id, value = beer_id.
    """
    for name in brewery.values():
        print(name)

def location_filter(ranked_beers, lookup_dict, state, city, n):
    """ 
    This takes a list of tuples where the 1st element is a beer_id. It searches through the lookup dictionary
    to match breweries based upon their location. And returns n number of recommendations

    It returns the beer_id as key, and brewery_name, beer id, and beer name as values
    """
    located_brewery = {}
    # state = 'CA'
    # city = 'Los Angeles'
    counter = 0

    for beer in ranked_beers:
        if counter < n:
            dict_state = lookup_dict[beer[0]]['state']
            dict_city = lookup_dict[beer[0]]['city']
            brewery_id = lookup_dict[beer[0]]['brewery_id']
            brewery_name = lookup_dict[beer[0]]['brewery_name']
            beer_name = lookup_dict[beer[0]]['name']
            if (dict_state == state) and (dict_city == city):
        #             print(beer_breweries_lookup[beer[0]])
                if brewery_id in located_brewery:
                    continue
                else:  
                    located_brewery[brewery_id] = (brewery_name, beer[0], beer_name)
                counter += 1
    return located_brewery

def location_filter2(ranked_beers, lookup_dict, state, city, n):
    """ 
    This takes a list of tuples where the 1st element is a beer_id. It searches through the lookup dictionary
    to match breweries based upon their location. And returns n number of recommendations

    It returns the beer_id as key, and brewery_name, beer id, and beer name as values
    """
    located_brewery = {}
    # state = 'CA'
    # city = 'Los Angeles'
    counter = 0

    for beer in ranked_beers:
        if counter < n:
            dict_state = lookup_dict[beer[0]]['state']
            dict_city = lookup_dict[beer[0]]['city']
            brewery_id = lookup_dict[beer[0]]['brewery_id']
            brewery_name = lookup_dict[beer[0]]['brewery_name']
            beer_name = lookup_dict[beer[0]]['name']
            if (len(state) > 0) and (len(city)>0):
                if (dict_state == state) and (dict_city == city):
            #             print(beer_breweries_lookup[beer[0]])
                    if brewery_id in located_brewery:
                        continue
                    else:  
                        located_brewery[brewery_id] = (brewery_name, beer[0], beer_name)
                
                    counter += 1
            # ignores state field
            elif len(state) == 0:
                if (dict_city == city):        
                    if brewery_id in located_brewery:
                        continue
                    else:  
                        located_brewery[brewery_id] = (brewery_name, beer[0], beer_name)
                
                    counter += 1

            elif len(city) == 0:        
                if (dict_state == state):
                    if brewery_id in located_brewery:
                        continue
                    else:  
                        located_brewery[brewery_id] = (brewery_name, beer[0], beer_name)
                
                    counter += 1
    if len(located_brewery) > 0:
        return located_brewery
    else:
        return 

def beer2beer(state, city, model, kw_or_beer):
    kw_or_beer = kw_or_beer.title()
    for i in lookup_dict:
        if lookup_dict[i]['name'] == kw_or_beer:
            recs = model.docvecs.most_similar(str(i), topn=10000)
            return location_filter2(recs, lookup_dict, state, city, 3)


def get_recs_from_wordvec(state, city, keyword, n_recs=3, topn=8000, stem=True):
    """
    takes in a word vec and returns top breweries from the provided location

    """
    if stem == True:
        ls = LancasterStemmer()
        model = load_alt_model()
        try:
            vec = model[ls.stem(keyword)]
            tags = model.docvecs.most_similar([vec], topn=topn)
            return location_filter2(tags, lookup_dict, state, city, n_recs)
        except KeyError:
            return

def user_recs(algo, reviews_df, uid):
    """
    Predicts ratings for each item for a given user id. Also takes the algorithm and 
    reviews dataframe

    'uid' is a string that is the username
    """
    
    list_of_beers = []
    for iid in reviews_df['id'].unique():
        list_of_beers.append((iid, algo.predict(uid,iid)[3]))
    ranked_beers = sorted(list_of_beers, key=lambda x:x[1], reverse=True)
    return ranked_beers


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.
    predictions is a df obtained by calling algo.test(testset)
    
    k is the top ranks predicted by the model
    
    threshold is the score value to determine if a predicted item is an item the user would like,
    the model compares those estimated and base truth scores to determine precision/recall of the 
    top k items
    '''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls

def recommendations(beer_id, cos_sim):
    """
    Takes a beer id and cosine similarty matrix in as arguments and returns beers closely related to the input beer
    """
    # initializing the empty list of recommended movies
    recommended_beers = []
    
    # gettin the index of the movie that matches the title
    idx = indices[indices == beer_id].index[0]
    print(idx)
    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cos_sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)
    print(top_10_indexes)
    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_beers.append(list(beers_text.name)[i])
        
    return recommended_beers

def tfidf_recs(beer_id, beer_df, cos_sim):
    """
    Takes a beer id and cosine similarty matrix in as arguments and returns beers 
    closely related to the input beer
    """
    indices = pd.Series(beer_df.index)

    # initializing the empty list of recommended movies
    recommended_beers = []
    
    # gettin the index of the movie that matches the title
    idx = indices[indices == beer_id].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cos_sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:21].index)
    
    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_beers.append(list(beers_text.name)[i])
        
    return beers_text.name[beer_id], recommended_beers

def combine_text_reviews(preprocessed_df):
    """
    This takes the preprocessed reviews and groups all of an individual beer's reviews into one feature
    for countvectorizer/tfidf models. It also removes some weird unicode.
    """
    # groups by beer id and joins all of their text reviews into a new feature: joined_text
    preprocessed_df['joined_text'] = preprocessed_df.groupby('id')['text'].transform(lambda x: ''.join(x))
    
    # drops columns that are not needed and also drops duplicate rows based on beer id
    preprocessed_df = preprocessed_df[['id', 'joined_text', 'avg_score', 'no_of_ratings']].drop_duplicates(\
                                                                        subset='id')
    # removes \xa0 remove text
    preprocessed_df['joined_text'] = preprocessed_df['joined_text'].apply(lambda x: re.sub\
                                                                        (r'\xa0', '', x))


def preprocess_reviews(reviews, beers):
    """
    this function comprises cleaning I went through during my initial EDA. I filter out beers that
    have no text reviews, are marked not retired, and I do some feature engineering (avg score and
    number of ratings for both beers and users). I lastly removed reviews whose average scores were
    outside of 2 STD of the average (only 1200 users). Doing so improved accuracy - many of these
    users were not performing as a 'normal' reviewer eg giving one brewery all 5s, or just reviewing
    all beers as 5. Some of these users also gave certain breweries bad reviews.
    """
    reviews['text'] = reviews['text'].replace(u'\xa0\xa0', '')
    # subset to only reviews that have a text review
    text_reviews = reviews.loc[reviews['text'] != '']
    # subset data to exclude NaN's as well (only losing 164k reviews from the last subset)
    text_no_nan = text_reviews.loc[text_reviews.smell.isna() == False]
    # rename column name beer_id to id for easy joining
    text_no_nan = text_no_nan.rename(columns={'beer_id':'id'})
    # subset out retired beers
    current_beers = beers.loc[beers['retired'] == 'f']
    # merge text_no_nan with beers that are not retired
    df = pd.merge(text_no_nan, current_beers, on='id')
    # create a table with average ratings for each beer. Index/ID is the beer id
    ratings = pd.DataFrame(df.groupby('id')['score'].mean())
    # add a column tallying the # of reviews for that beer
    ratings['no_of_ratings'] = df.groupby('id')['score'].count()
    # subset ratings with only beers that have 10+ ratings
    ratings = ratings.loc[ratings['no_of_ratings'] > 9]
    # formatting
    ratings = ratings.reset_index()
    ratings = ratings.rename(columns={'score':'avg_score'})
    # rejoin no of ratings onto df
    df = df.merge(ratings, how='inner', on='id')
    # make a dataframe of reviewers by usename, count the number of reviews they made
    reviewers = pd.DataFrame(df.groupby('username')['id'].count())
    # make a new feature, the average of all of their scores
    reviewers['avg_usr_score'] = df.groupby('username')['score'].mean()
    # subset reviewers to those with 5+ reviews. From 73k users to 25k.
    reviewers = reviewers.loc[reviewers['id'] > 4] ## MAYBE I CAN PLAY WITH THIS #
    # formatting 
    reviewers = reviewers.rename(columns={'id':'tot_usr_rvw'})
    # there's only ~1400 users outsides of 2 STDs of the mean score, will subset them out
    reviewers_sub = reviewers.loc[(reviewers['avg_usr_score'] >= 3.182) &\
                                (reviewers['avg_usr_score'] <= 4.665)]

    # subset of df with beers that have 10+ reviews, and with reviewers that have 5+ reviews
    # and an average rating of beers between 3.18 and 4.67
    df_with_mins = df.merge(reviewers_sub, how = 'inner', on = 'username')

    return df_with_mins

def get_user_pred_set(user_rating_list, rating_df):
    """returns a list of beer id's to be predicted. excludes beers the users imputed
    user_rating_list: is a list of dictionaries produced when the user provides
                      initial ratings
    rating_df: is a df of all ratings subseted to the columns needed for SVD"""
    
    user_ratings_ids = []
    for rating in user_rating_list:
        user_ratings_ids.append(rating['id'])
    beers_for_pred = []
    for beer_id in rating_df['id']:
        if beer_id not in user_ratings_ids:
            beers_for_pred.append(str(beer_id))
    return set(beers_for_pred)

def svd_location_filter(user_pred_list, lookup_dict, state, city, n):
    """ 
    takes in list from get_user_pred_list and filters list down to only the location they
    provided, returns a dictionary with the beer id as key, and beer_name and brewery_name
    as values
    """
    located_beer = {}
    counter = 0

    for beer in user_pred_list:
#         print(beer)
        if counter < n:
            dict_state = lookup_dict[beer]['state']
            dict_city = lookup_dict[beer]['city']
            brewery_id = lookup_dict[beer]['brewery_id']
            brewery_name = lookup_dict[beer]['brewery_name']
            beer_name = lookup_dict[beer]['name']
            if (dict_state == state) and (dict_city == city):
        #             print(beer_breweries_lookup[beer[0]])
                if brewery_id in located_beer:
                    continue
                else:  
                    located_beer[beer] = (beer_name,brewery_name)
                counter += 1
    return located_beer

def sort_score(val):
    """used to sort predictions by their estimated score"""
    return val[1]

def pred_for_user_location(to_predict_list, username, model):
    """Takes in a list of beer ID's (that have been filtered by location) to be predicted for 
    the given user. Also takes the trained model as an argument"""
    predictions = []
    for iid in to_predict_list:
        pred = model.predict(username, int(iid), verbose = False)
        predictions.append((pred[1],pred[3]))
    predictions.sort(key = sort_score, reverse = True)
    return predictions

def return_top_breweries(top_beers, svd_loc_filter_output, n):
    
    counter = 0
    top_breweries = {}
    for beer in top_beers:
        if counter < n:
            beer_name = svd_loc_filter_output[str(beer[0])][0]
            brewery_name = svd_loc_filter_output[str(beer[0])][1]
            if brewery_name in top_breweries:
                continue
            else:
                top_breweries[brewery_name] = (brewery_name, beer_name)
#             top_breweries.append(svd_loc_filter_output[str(beer[0])])
            
            
            counter+=1
    return top_breweries
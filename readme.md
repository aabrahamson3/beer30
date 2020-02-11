## Beer30 - A Brewery Recommender for Travellers powered by deep NLP

The craft beer industry is one of the fastest growing in the US. From 2017 to 2018 alone there was a 15% increase in microbreweries, from 2014 to 2018 there has been a 117% increase. As someone that enjoys both travel and great craft beer it is overwhelming going to a new city and trying to figure out what breweries I should visit. This project was created to help people (and me) discover what breweries they should visit while in a new town.

The project utilizes this dataset (https://www.kaggle.com/ehallmar/beers-breweries-and-beer-reviews) from Kaggle. It is comprised of over 9 million user reviews spanning the years 1998-2017 from www.BeerAdvocate.com. 

Here is some other data and notes after performing my EDA (the notebook is in this repo):
- Initially I attempted to create a collaborative filtering system, but it proved to have issues recommending unique beers. The final model is a content based filtering system utilizing the text reviews
- After subsetting the data to relevant reviews with text, the final dataset used by the project is just under 2 million text reviews on 22,000 unique beers

The model uses gensim's Doc2Vec, and in this context each beer is a unique document. The user is able to either input a beer they like (and the model searches for similar document) or a descriptive word (and the model find documents closest to that word vector).

In its current state there is not a way to quantitatively evaluate the performance of the model. A future goal would be to include a way for the user to indicate if the recommendations were accurate, or not.

The model is deployed via a Flask app here: https://tinyurl.com/beer30rec

____________________________________________________________________________________________________________

Instructions:

The dataset is too large to be included on this repo, but there are cleaning and preprocessing functions in the functions.py file. Required packages are included in the environment.yml file.

Below are the notebooks and their contents:
- Beer-EDA: Contains initial EDA and data cleaning
- 1.0-Recommender: Contains initial work with a collaborative filter using Surprise and PySpark
- Doc2Vec: Contains work with Doc2Vec model. Model training and evaluation, and t-SNE visualization


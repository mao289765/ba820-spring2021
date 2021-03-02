##############################################################################
## Foundations in Text Analytics - Sentiment Analysis
##
## Learning goals:
##                 - reinforce text as a datasource
##                 - python packages for handling our corpus for these specific tasks
##                 - Sentiment analysis 
##                 - Sentiment analysis via ML
##                 - Build your own sentiment classifier!
##############################################################################

# installs
# ! pip install newspaper3k
# ! pip install spacy
# ! pip install wordcloud
# ! pip install emoji
# ! pip install nltk
# ! pip install scikit-plot
# ! pip install umap-learn
# ! pip install afinn
# ! pip install textblob

# imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplot

# some "fun" packages
from wordcloud import WordCloud
import emoji

import re

# text imports

import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer  
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from afinn import Afinn

from newspaper import Article



#### the last exercise before break

afinn = Afinn()

SQL = "SELECT * from `questrom.datasets.bruins_twitter` where hour IN (0,1,2,3)"
PROJ = "questrom"
tweets = pd.read_gbq(SQL, PROJ)

tweets.shape
tweets.head()

# bonus, if you wanted to parse the created at
tweets['created'] = pd.to_datetime(tweets.created_at)

# a function that we can apply
def sent_score(text):
  return afinn.score(text)

tweets['sent'] = tweets.text.apply(sent_score)

# with status id, its somewhat like a timeseries
tweets.sort_values("status_id", ascending=True, inplace=True)
tweets.reset_index(drop=True, inplace=True)

tweets.sent.plot(kind="line")
plt.show()


# average by hour -- refer to 7 script for renaming cols if you like
tweets.groupby("hour").agg({'sent':['size','mean']})


###############################  lets look at two records

IDS = [1204921609733070848, 1206041140165849089]
# sent = 0,-1
tweets.loc[tweets.status_id.isin(IDS), "text"].values
tweets.loc[tweets.status_id.isin(IDS), :]

################## WHAT DO YOU NOTICE ABOVE
################## should these be slightly neutral or negative?

afinn.score("empty")

##################################### Quick departure
## word clouds
## you may have clients ask about this
## 
## let's break this down
##

# we will use the article text from here
# expects a string, more or less, not a list, per se

URL = "https://bit.ly/37ShEj4"
article = Article(URL)
article.download()
article.parse()
article.text

wc = WordCloud(background_color="white")
wordcloud = wc.generate(article.text)

# # Display the  plot:
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

#### run above a few times, look for
## 1- the color of GPT, medical, and expo
## 2- the placement relative to each other

################################## THOUGHT EXPERIMENT/REVIEW OF ABOVE
##
## Ok, let's break this down
## we have seen distance measures + PCA/TSNE/UMPA
## in those situations, spatial placement matters
## general best practices - color should mean something too
## size in word clouds is fine
## Takeaway:  this is really, really "fun" way to explore data
##            but hardly one that should ever be considered anlaytics or findings (IN MY OPINION)
##
## WHY?   How does the shape of sunglasses help us understand the data? LOL
## https://www.littlemissdata.com/blog/wordclouds
##

##################################### TextBlob - Sentiment
## 
## NLP toolkit that goes beyond sentiment, so worth exploring
## we will see spacy on Wednesday, so this is a nice segue
## 
## The general framework: we operate on a document, not a corpus
## the document is parsed for a variety of NLP tasks
##
## POS, sentiment, noun-phrases, even spelling correction
##
##

# setup
from textblob import TextBlob

# a simple corpus to show the document orientation
corpus = ["Today was a great day", "Today was a bad day"]

# attempt to parse, this will fail!
# TextBlob(corpus)

## REVIEW THE TypeError

# lets try this again

parsed = []
for doc in corpus:
  parsed.append(TextBlob(doc))

# we have the objects "parsed", 
# we can see that the data are TextBlob objects
type(parsed[0])

# what do we have, look at the first

# lets get the sentiment
sent = []
for p in parsed:
  sent.append(p.sentiment)

sent

polarity = []
for doc in corpus:
    polarity.append(TextBlob(doc).sentiment.polarity)

polarity


############################### REVIEW
# we can see above that we are still getting the objects, but we start to see some detail
#
# polarity = how positive/negative, which ranges from -1 to 1
# subjectivity = 0 -> 1, or fact -> opinion
#
# not shown above, but textblob's model also handles modifier words which we will see in a moment
#
# ALSO: the model?  Trained on reviews that were labeled (movie reviews), and also draws from other projects
#                   https://raw.githubusercontent.com/aesuli/SentiWordNet/master/data/SentiWordNet_3.0.0.txt
#                   https://github.com/sloria/TextBlob/blob/90cc87ab0f9e25f37379079840ec43aba59af440/textblob/en/sentiments.py
# Link below you can dive into the code for more view
#                   https://github.com/sloria/TextBlob/blob/eb08c120d364e908646731d60b4e4c6c1712ff63/textblob/_text.py

# lets go back to the results though, we can pull out the scores as needed

sent = []
for doc in corpus:
  sent.append(TextBlob(doc).sentiment.polarity)

sent

# NOTE:  that unlike afinn, these are not balanced, but perhaps because there is more than
# just a lookup

# lets try some other examples
# exclamation and negation are factors, but all caps is not

TextBlob("not great").sentiment.polarity
afinn.score("not great")

TextBlob("very awesome").sentiment.polarity
TextBlob("not awesome!").sentiment.polarity



###### you can see above that there are some smarts baked into this
##     textblob is able to look at the words and modify as needed, based on the intensity of the word (not shown)
#      some punct rules and negation

# one other note
# Textblob ignores 1 letter words, just like sklearn does

# subjectivity?

TextBlob("python is awesome").sentiment.subjectivity

##################################### Hands on
##
##  We will use the same bruins twitter dataset above
##  refer to above if you want to re-query the data
##  
## 
##  calculate the polarity (sentiment) and subjectivity for each tweet
##  create a scatterplot to evaluate the both metrics for the dataset
##
##   next plot the relationship between afinn score and textblob score
##

def tb_p(text):
  p = TextBlob(text).sentiment.polarity
  return p

def tb_s(text):
  s = TextBlob(text).sentiment.subjectivity
  return s

tweets['polarity'] = tweets.text.apply(tb_p)
tweets['subjectivity'] = tweets.text.apply(tb_s)

tweets.head(3)

# polarity vs subjectivity
sns.scatterplot(x="polarity", y="subjectivity", data=tweets)
plt.show()

# affinn sent versus textblob polarity
# sns.scatterplot(x="polarity", y="sent", data=tweets, alpha=.5)
sns.lmplot(x="polarity", y="sent", data=tweets)
plt.show()


##################################### Vader sentiment
## Very brief review
## There are other approaches that attempt the modified approach
## you should inspect them at a granular level to ensure that these work as you like
## 
## But for sake of completeness, lets see the output
##
## for a deeper review on compound scores
## https://stackoverflow.com/questions/40325980/how-is-the-vader-compound-polarity-score-calculated-in-python-nltk
##  -- sum of normalized scores, and is not directly related to the pos/negative/netural
##  https://github.com/cjhutto/vaderSentiment#about-the-scoring
##
## we get pos/neutral/negative distros 
## and a compound score which is what mostly used in practice
## > .05 = positive
## < -.05 = negative
## else neutral
## 
## like Textblob, its a model with modifiers and intensifiers
## but this model was trained on social media, so perhaps it may fit your data better
## AGAIN:  these tools help us get a report off the ground quickly, but we always should review
##

# vader installs via nltk

nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

# init

vader = SentimentIntensityAnalyzer()

# lets play around -- whats nice is the distro can be a plot you use to highlight how this is subjective

vader.polarity_scores("Today is a great day")
vader.polarity_scores("Today is a bad day")
vader.polarity_scores("cool")
vader.polarity_scores("very cool")
vader.polarity_scores("not very cool")
vader.polarity_scores("great day")
vader.polarity_scores("great day!")
vader.polarity_scores("GREAT day")

vader.polarity_scores("That movie was so bad, but that's why I like it")

# but does punctuation matter?  how about ALL CAPS?

# corpus = ["today was a good day", "today was a good day!"]
# test = []
# for doc in corpus:
#   test.append(vader.polarity_scores(doc))

# test

# and if you wanted to look at the tweets
# this might take a few moments!

def vader_model(text):
  vader_score = SentimentIntensityAnalyzer()
  sent = vader_score.polarity_scores(text)
  return sent.get("compound", 0)

tweets['vader'] = tweets.text.apply(vader_model)

# a few records

## lets go back to those ids from before!

tweets.loc[tweets.status_id.isin(IDS), :]

###################### above highlights that we might need to build our own!

##################################### ML approach
## 
## Sometimes we need to roll our own
## 
## What does this mean?
## 1. collect a dataset
## 2. annotate the data with our own business rules
##  --------> Label studio?
## 3. we can use some of the tools above 
## ---------> generate a score, define a threshold, give labels
## ---------> fit a model on labels
## ---------> review, iterate, review, iterate
##
## Why build our own
## 
## - out of the box generalize (thats a theme you have heard me say)
## - domain specific words (for example, dataset above shows sports terms that are positive but not captured
## - also, sarcasm is hard to detect even with modifier approaches
##
##

# there is an airlines tweets dataset on biq query
# bring in questrom.datasets.airlines-tweets
# just the tweet_id, airline_sentiment, airline, and text columns

SQL = """select tweet_id, 
                airline_sentiment, 
                airline, 
                text 
         from `questrom.datasets.airlines-tweets`
"""
airlines = pd.read_gbq(SQL, "questrom")

# shape and a few records
airlines.shape
airlines.head(3)


# what do we have for a label distribution?

airlines.airline_sentiment.value_counts(normalize=True)


# lets assume our excellent back office staff has labeled these datasets properly
# huge assumption, right!
# we will parse the tweets, convert emojis, keep top 1000 vocab
# and then create our own ML-based sentiment classifier

# example of demojizing a text - just parses them out!

txt = "great , but I have to go with #CarrieUnderwood 😍👌"
emoji.demojize(txt)

# lets setup the flow of a model 

from nltk.tokenize import TweetTokenizer
import emoji

SW = stopwords.words('english')

# parse the airline tweets into a ML dataset

def tokenizer(text):
  social = TweetTokenizer()
  # replace emojis with string representations
  text = emoji.demojize(text)
  # if two emojis are stacked, add a whitespace in between
  text = text.replace("::", ": :")
  return social.tokenize(text)

cv = CountVectorizer(tokenizer=tokenizer, stop_words=SW, max_features=1000)

tokens = cv.fit_transform(airlines.text)


# into a dataframe

adf = pd.DataFrame(tokens.toarray(), columns=cv.get_feature_names())

# view a few entries with .iloc
adf.iloc[:5, :5]

# top tokens

adf.sum(axis=0).sort_values(ascending=False)

############# notice above, what could we do better?

# regardless lets fit a decision tree classifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

tree = DecisionTreeClassifier(max_depth=5, min_samples_split=40)

# validation

X = tokens.toarray()
y = airlines.airline_sentiment

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, stratify=y)

# fit the tree

tree.fit(X_train, y_train)

# apply the tree

preds = tree.predict(X_test)

# the report

from sklearn.metrics import classification_report

cr = classification_report(y_test, preds)
print(cr)

############################ THOUGHT/PRACTICE EXERCISE
## how might you improve this?
## this is good practice!
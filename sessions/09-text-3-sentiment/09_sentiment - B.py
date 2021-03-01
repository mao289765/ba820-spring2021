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

##################################### let's get warmed up
## URL = https://bit.ly/37ShEj4
## 
## scrape the text from the URL above
## tokenize based on word characters
## TRICKY - tokenize based on word characters and a hyphen (-)
## include unigrams and bigrams
## remove stopwords
## how large is the vocab
## TRICKY: top 5 tokens
##
##


URL = "https://bit.ly/37ShEj4"
article = Article(URL)
article.download()
article.parse()
article.text

# stopwords
SW = stopwords.words('english')

# vectorizer
cv = CountVectorizer(token_pattern="[\w-]+", stop_words=SW, ngram_range=(1,2))

# tokens
tokens = cv.fit_transform([article.text])

# length of the vocabulary
len(cv.vocabulary_)

# top 5 
df = pd.DataFrame(tokens.toarray(), columns=cv.get_feature_names())
df.shape

vocab = df.sum(axis=0).sort_values(ascending=False)
vocab[:5]



##################################### Sentiment 1
##
##  We will start basic = word/dictionary-based approach
## 
## IDEA:  each word gets a score, sum up the score, thats it!
## it's intuitive, easy to explain, and customizable
## 
## Afinn - 2011
## https://github.com/fnielsen/afinn
##
## TLDR
## limited language support, but highlights an important concept
## build our dictionary, and score
## could even be emoticons!
## https://github.com/fnielsen/afinn/tree/master/afinn/data
##
##

# setup the english "model"
afinn = Afinn(language='en')

# let's just start with something basic
afinn.score("Today is a great day!")

# let's try another
afinn.score("Today is a bad day!")

############### Question:  What do you notice?  What happened (outside of getting a score)?

## let's look at the data behind this
afinn.score("good")
afinn.score("bad")


URL = "https://raw.githubusercontent.com/fnielsen/afinn/master/afinn/data/AFINN-111.txt"
ad = pd.read_csv(URL, sep='\t', header=None, names=['token', 'score'])
ad.sample(7)

# summary -- bounded between -5 and 5
##  mean is slightly negative (negative values = negative)
ad.score.describe()


# what are the values for score
sns.distplot(ad.score)
plt.show()

# we can inspect easily to wrap our heads around the words
# stop

ROWS = ad.token.str.contains("stop")
ad.loc[ROWS, :]

# another search - lol
ROWS = ad.token.str.contains("lol")
ad.loc[ROWS, :]


# let's go back to a statement, see the score, and break it down

doc = "love hate"
afinn.score(doc)

# what is the list of floats being used for the score

afinn.scores(doc)

# confirming that we can make ths more concrete, and that
# the other words are not considered important for sentiment
# in a lookup approach

afinn.scores("love hate relationship with this service")

doc = "Today is a good day!"
afinn.score(doc)

doc = "Today is not a good day!"
afinn.score(doc)



##################################### YOUR TURN
##
##  there is a table on big query
##  datasets.bruins_twitter
##
##  get the records where the hour is 0,1,2,3
##  this is not a select *, you have to filter records
##  apply afinn sentiment to each record
##  ensure that the data sorted by status_id
##  plot the sentiment score over the records (this is a timeseries - like view)
##  calculate the average sentiment by hour
##
##




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


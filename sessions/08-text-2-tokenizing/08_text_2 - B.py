# LEARNING GOALS
#
#                 - tokenization deeper dive
#                 - introduce ML integrations
#                 - reinforce text prep
#                 - Cluster documents

# installs
# ! pip install newspaper3k
# ! pip install spacy
# ! pip install wordcloud
# ! pip install emoji
# ! pip install nltk
# ! pip install scikit-plot
# ! pip install umap-learn

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

# new imports
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer,TfidfVectorizer  
import nltk

from newspaper import Article

# lets go back to our simple dataset
a = ['I like turtles!',
     'You like hockey and golf ',
     'Turtles and hockey ftw',
     'Python is very easy to learn. 🐍',
     'A great resource is www.spacy.io',
     ' Today is the Feb 22, 2021 !           ',
     '@username #hashtag https://www.text.com',
     'BA820 ',
     'My name is Brock and my phone number is 617-867-5309']

df = pd.DataFrame({'text':a})
df

# remember, the printout is cleaned up!
df.text.values

# what we saw previously was the intuition of tokenzing our data
# but its not hard to imagine that there has been a great deal of work towards this task
# let's start with just getting the 

# first, lets use sklearn
cv = CountVectorizer()
tokens = cv.fit_transform(a)

# what do we have?
type(tokens)

# shape?
tokens.shape

# we get a sparse array which is nice, but we can "expand" this
tokens.toarray()


# confirm the length matches
len(a) == len(tokens.toarray())

# this is a bag of words approach
# one row for the document, and a count vector for each token
#
# NOTE:  sklearn is tokenizing the words for us, but we will come back to this

# what's nice about sklearn is that it keeps things simple and retains
# the nice aspects of working in this toolkit
cv.vocabulary_


# get the features and the index

# we can also extract the feature names
cv.get_feature_names()

## when considering the original input, what stands out to you?

# you can always just pull out the feature names if you want
# but the goal here is that sklearn is keeping this to help with downstream ml

############################################ your turn
## URL = https://voicebot.ai/2021/02/16/conversational-ai-startup-admithub-raises-14m-for-higher-ed-chatbots/
## get the page and extract the text  (HINT: newspaper or requests/beautifulsoup can help!)
## tokenize the page (CountVectorizer)
## how many tokens do you have?
## TRICKY!  which word appears the most often?  what is it's index?
## TIP: may need to pass to sklearn as a list of length 1

## REMEMBER: a few ways to do this, but always can copy/paste if you don't want to scrape with newspaper or requests/soup


URL = "https://voicebot.ai/2021/02/16/conversational-ai-startup-admithub-raises-14m-for-higher-ed-chatbots/"

article = Article(URL)
article.download()
article.parse()
article.text

cv = CountVectorizer()
atokens = cv.fit_transform([article.text])
atokens.shape

cv.get_feature_names()

# matrix
tmat = atokens.toarray()
tmat.shape

summary = np.sum(tmat, axis=0)
token1 = np.argmax(summary)
cv.get_feature_names()[token1]




################################################## Lets summarize so far
##
## we can use sklearn to keep things in our typical ml format
## we can see that there is some pre-processing taking place
## lets dive into that a bit more, and then discuss a flow using nltk -> sklearn

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

# - Notice the lower casing by default
# - we can pass our own regex/tokenizer if we wanted, and some people do this (build their own)
# - different ways to tokenize
# - there are stopwords, but we can pass anything
# - we can set the max number of tokens
# - we can one hot encode = instead of counts, it can be 0/1 for the word/token
# - we can create ngrams
# - we can even validate the vocabulary if we wanted
#
# This last point brings up the concept of unseen words
# Remember! sklearn fits the object, so any unseen words will not be parsed on new datsets with transform
#
# Summary: really powerful and adaptable, but means you plug in your own regex/tools

################### part 1: - lets start with ngrams
##
## instead of single tokens, we can try to capture context by windowing the tokens/phrases
## we can pass in a tuple of the ngrams, default is 1,1

# a new dataset
corpus = ["tokens, tokens everywhere"]

# we could only have bigrams
ngram2 = CountVectorizer(ngram_range=(3,3))
ngram_tokens = ngram2.fit_transform(corpus)
ngram2.get_feature_names()
len(ngram2.get_feature_names())

# the key point is that you can imagine it might be able to retain context
# if we combine tokens with other n-grams.  
#

###################################### Quick task
## 
## build off the article from above
## but instead of parsing the tokens (unigrams), include bigrams (2) and trigrams (3) 
## to the feature space
##
## how many features have we extracted from the article?
##

###################################### Question
###### what does this say about our choice of tokenization
###### what tools might help with this "issue"?

###################################### Stopwords
## by default stop words are not removed
## there is a pre-built list of words, but let's ignore it
## nltk is a great toolkit, and we will explore it later, but for now
## lets just use the stopwords from that package

# if this is your first time, you may need to download the stopwords
# or on colab, for your session
nltk.download('stopwords')


## OF COURSE, you could always downlod your own.  not the format of below, we just pass in a list in the end

# lets get the stopwords
from nltk.corpus import stopwords
STOPWORDS = list(stopwords.words('english'))

# what do we have?
type(STOPWORDS)

# the first few
len(STOPWORDS)

# note that everything is lower case!
STOPWORDS[:3]

import random
random.choices(STOPWORDS, k=10)

# admittedly this is harder to find than it should be
# but the languages supported in NLTK
stopwords.fileids()

# now you can imagine that is pretty limiting above, I know
# the other approach is to use spacy
# https://spacy.io/usage/models
# we will dive into spacy later, but I think its important to keep building the intuition
# before going into model-driven work

# last, we can always add to the stoplist if we wanted to now that its a list abvoe

# lets keep the corpus small, so use the original 
# but remove stopwords
cv = CountVectorizer(stop_words=STOPWORDS, ngram_range=(3,3))
atokens = cv.fit_transform(a)
len(cv.get_feature_names())


# 37 -> 30

# and of course, we can see the vocab
len(cv.get_feature_names())

df = pd.DataFrame(atokens.toarray(), columns=cv.get_feature_names())
df
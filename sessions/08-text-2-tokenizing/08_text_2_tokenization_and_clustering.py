# LEARNING GOALS
#
#                 - tokenization deeper dive
#                 - introduce ML integrations
#                 - reinforce text prep
#                 - Cluster documents

# installs
! pip install newspaper3k
! pip install spacy
! pip install wordcloud
! pip install emoji
! pip install nltk
! pip install scikit-plot
! pip install umap-learn

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

# shape?

# we get a sparse array which is nice, but we can "expand" this

# confirm the length matches

# this is a bag of words approach
# one row for the document, and a count vector for each token
#
# NOTE:  sklearn is tokenizing the words for us, but we will come back to this

# what's nice about sklearn is that it keeps things simple and retains
# the nice aspects of working in this toolkit

# get the features and the index

# we can also extract the feature names

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

# the first few

# note that everything is lower case!

# import random
# random.choices(STOPWORDS, k=10)

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
cv = CountVectorizer(stop_words=STOPWORDS)



# 37 -> 30

# and of course, we can see the vocab

###################################### Max tokens
## 
## this can be helpful if you want to restrict to the top N most frequent tokens
## this restricts your space at the start
## but the tradeoff is less common words, perhaps, could help with ML models
##     -- the tokens/phrases are specific to he known label, and while rare, often occur for the label

## we can use the article again, max with stopwords

# cv = CountVectorizer(max_features=20, stop_words=STOPWORDS)
# atokens = cv.fit_transform([atext])
# cv.vocabulary_

###################################### character tokens
## 
## if you wanted, you can parse characters
## a little out of scope, but highlighting the concept of tokenization can 
## take all sorts of forms!

x = ["Hello I can't"]

# charvec = CountVectorizer(analyzer='char', ngram_range=(1,1))
# cv = charvec.fit_transform(x)

###################################### custom pattern
## 
## Finally, if you really wanted to (or needed to), you can roll your own
## tokenization
## This is a little forward looking  .....
## but highlights you all have the power to roll your own
##
## https://stackoverflow.com/questions/1576789/in-regex-what-does-w-mean
##

# alpha numeric plus a single quote/contraction
PATTERN = "[\w']+"

cv = CountVectorizer(token_pattern=PATTERN)

###################################### Your Turn
## 
## get the text from the two articles below
## 1.  https://towardsdatascience.com/can-we-please-stop-using-word-clouds-eca2bbda7b9d
## 2.  https://www.businessinsider.com/pie-charts-are-the-worst-2013-6
##
## create a bag of words representation of the two documents
## keep the top 250 word tokens
## remove stopwords
## use tokens, bigrams (2) and trigrams(3)
## TRICKY!  Put back into a dataframe if you can

# a1 = Article("https://towardsdatascience.com/can-we-please-stop-using-word-clouds-eca2bbda7b9d")
# a1.download()
# a1.parse()
# a2 = Article("https://www.businessinsider.com/pie-charts-are-the-worst-2013-6")
# a2.download()
# a2.parse()

## remember, jaccard is intersection over union, 
## instead of counts, we just said "is this word present"
## value is proportion of elements that disagree

## lets do a little more parsing before we start clustering!

############################################################
########################################### Team Challenge
############################################################

## Get a first pass model built to predict if an SMS message is spam!
## 
## review the slides at the end of this module
## predict spam based on the message
## objective =  based on accuracy
## 
## only input is text, but you can derive features if you like
## limited time, but how do you maximize your initial first pass (and the model?)
## HINTS:
##        start small, simple models
##        iterate and see how you do against the leaderboard
##        code above helps you with the core mechanics

###################################### NLTK parsing
###################################### Quick highlight that there are pre-built tools!
## 
## we may not have to reinvent the wheel!
## NLTK has some built in tooling we can leverage!
## and trust me, other toolkits have their own approaches too!

from nltk.tokenize import word_tokenize, RegexpTokenizer, WordPunctTokenizer, TweetTokenizer

# we may also need to download a tool to help with (sentence) parsing amongst other tasks
nltk.download('punkt')

corpus = ['I want my MTV!', "Can't I have it all for $5.00 @customerservice #help"]

## first thing: simple word tokenize

# tokens = []
# for doc in corpus:
#   tokens.append(word_tokenize(doc))

## what do you notice?

# lets dive deeper
wp = WordPunctTokenizer()




# even more granular split



# just like scikit, roll our own
# basically split on whitespace by NOT (uppercase) selecting whitespace

# note we have to add our pattern below
regtok = RegexpTokenizer()




# strips out other bits

# finally, a tokenizer to help with twitter, and perhaps other social data
social = TweetTokenizer()




# what do we have

###################################### Summary
## 
## we have super powers via regex, but don't be afraid to look around
## some decent tools in sklearn, but nltk has some custom utilities we can leverage
##
## We have options!  
## we can try to parse with nltk and feed to sklearn
## we can use the tooling in sklearn but might require we roll our own modifications
##
## but generally the flow is pre/tokenize -> bag of words of those tokens
##

corpus

############################### So the big question
## how does this all fit together?

# build a function to pull in the bits we want from NLTK, or whatever framework we want to use
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return tokens

# NOTE: lower case happens below, not above



###################################### Next Up:  Beyond simple counts with TFIDF
##
## instead of count vectors (which you can use, and should try, in your modeling!)
## we can try to de-prioritize common words 
## This surfaces words that may be less common, but nuanced and we want to prioritize those tokens
##

# the same data
corpus = ["Can't I have it all for $5.00 @customerservice #help", 
          'I want my MTV!']

# equivalent to CountVectorizer -> TfidfTransformer
# basically if you want tfidf, do this, it saves a step
# and you have the same options for parsing if you like

## just to call out, being able to specify the pattern can be 
## really powerful for specific tasks and business needs

# lets put this into a dataframe



# we could even heatmap this to help understand the intuition here

# plt.figure(figsize=(4,6))
# sns.heatmap(idf.T, xticklabels=True, yticklabels=True, cmap='Reds')
# plt.show()

################### NOTE:
## look at the weights generally, what do you see?
## now focus in on the word in common, the token i
##
## we can see that when shared, there is the document effect

# https://towardsdatascience.com/a-gentle-introduction-to-calculating-the-tf-idf-values-9e391f8a13e5

# lets break the intuition down

# first the scores
# ivals = idf[["i"]]
# print(ivals)

## now just look at the statements, focus on i

## ^^ above, you can see that i is only 1 of 4 words
##    it highlights how its natrual to assume i, in that context, has a higher weight 
##

## on more try to extend this

# corpus2 = corpus.copy()
# corpus2.extend(["I want help", "halp", "python is fun", "Can I get help"])

# refit - but simplify, just tokens (unigrams)

# tfidf2 = TfidfVectorizer(token_pattern="[\w']+", ngram_range=(1,1))
# tfidf2.fit(corpus2)
# idf2 = tfidf2.transform(corpus2)
# idf2 = pd.DataFrame(idf2.toarray(), columns=tfidf2.get_feature_names())
# plt.figure(figsize=(4,6))
# sns.heatmap(idf2.T, xticklabels=True, yticklabels=True, cmap="Reds")
# plt.show()

## again could imagine that we might want to tighten up parsing of amounts
## but this is starting to get us to think about NER later own this semester
## for now, think of entity extraction as regex pattern searches


## but why does this matter?
##  We can think of tfidf as attempting to create a more informative feature space
##  when we think about similiarty, or how we could reduce this space easily, 
##  its not hard to consider that DR techniques give us ways to compress but 
##  we lose the ability to describe the impact of a given token.

# its easy to imagine this as a large space if we open up how we tokenize + ngrams

# lets tsne the data and plot the docs
# our data are so small that we dont need something like PCA in front

# from sklearn.manifold import TSNE

# tsne = TSNE()
# t2 = tsne.fit_transform(idf2)

# we just compressed the tfidf array to 2d

# lets plot

# now lets fit a kmeans to the dataframe
# what falls out with k = 2?
# OVERLY simplistic of course, but the hardest part was getting the feature vectors setup via tokenization
# obviously we have some flexibility on how we tokenize, the ngrams, stopwords, max threshold, etc
# so solutions can take different forms!

## if you wanted to see a different approach, use the count vectors
# cv = CountVectorizer(token_pattern="[\w']+", ngram_range=(1,1))
# cv.fit(corpus2)

# idf_c = cv.transform(corpus2)
# idf_c = pd.DataFrame(idf_c.toarray(), columns=cv.get_feature_names())
# plt.figure(figsize=(4,6))
# sns.heatmap(idf_c.T, xticklabels=True, yticklabels=True, cmap="Reds")

###################################### SUMMARY
##

###################################### BREAKOUT Challenge
##
## Get the topics from big query
## questrom:datasets.topics
## parse the text into bag of words
## (only the text, not the category) - your choice on tokenization and weighting/feature space
## cluster the text
## how many clusters do you have?
## overlay the category on top of the clusters
## if we didn't have the category, any evidence that  text processing and clustering would help
## find patterns?  Are there documents that appear to be outliers?



###################################### whats next?
##
## sentiment analysis- the easy vs the good (in my opinion, of course)
## review how/why this can work, and why sentiment is easy to do poorly
## continue to see how text and machine learning fit very well together
## spacy to get us thinking about parsing the entities, and gensim preview
##
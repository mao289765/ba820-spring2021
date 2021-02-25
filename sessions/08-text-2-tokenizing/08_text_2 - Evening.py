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
from pandas.tseries.offsets import CustomBusinessMonthBegin
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

cv = CountVectorizer()
tokens = cv.fit_transform(a)
cv.vocabulary_
cv.get_feature_names()
tokens.toarray()

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

cv = CountVectorizer(max_features=10)
tokens = cv.fit_transform(a)
cv.get_feature_names()

len(cv.get_feature_names())


cv = CountVectorizer(max_features=10, ngram_range=(1,4))
tokens = cv.fit_transform(a)
cv.get_feature_names()

len(cv.get_feature_names())



###################################### character tokens
## 
## if you wanted, you can parse characters
## a little out of scope, but highlighting the concept of tokenization can 
## take all sorts of forms!

x = ["Hello I can't"]

charvec = CountVectorizer(analyzer='char', ngram_range=(2,2))
cv = charvec.fit_transform(x)
charvec.get_feature_names()


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
cv.fit(a)
cv.get_feature_names()

cv = CountVectorizer(token_pattern=PATTERN)
cv.fit(x)
cv.get_feature_names()



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

a1 = Article("https://towardsdatascience.com/can-we-please-stop-using-word-clouds-eca2bbda7b9d")
a1.download()
a1.parse()
a2 = Article("https://www.businessinsider.com/pie-charts-are-the-worst-2013-6")
a2.download()
a2.parse()

from nltk.corpus import stopwords
STOPWORDS = list(stopwords.words('english'))

cv = CountVectorizer(max_features=250, stop_words=STOPWORDS, ngram_range=(1,3), binary=True)
cv.fit([a1.text, a2.text])
len(cv.get_feature_names())

bow_df = pd.DataFrame(cv.transform([a1.text, a2.text]).toarray(), columns=cv.get_feature_names())
bow_df.iloc[:5, :5]

from scipy.spatial.distance import pdist, squareform
dist = pdist(bow_df, metric="jaccard")
squareform(dist)
dist


## remember, jaccard is intersection over union, 
## instead of counts, we just said "is this word present"
## value is proportion of elements that disagree

## lets do a little more parsing before we start clustering!



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

tokens = []
for doc in corpus:
  tokens.append(word_tokenize(doc))
tokens

## what do you notice?

# lets dive deeper
wp = WordPunctTokenizer()

tokens_wp = []
for doc in corpus:
  tokens_wp.append(wp.tokenize(doc))

# even more granular split
tokens_wp


# even more granular split



# just like scikit, roll our own
# basically split on whitespace by NOT (uppercase) selecting whitespace

# note we have to add our pattern below
regtok = RegexpTokenizer("\S+")

tokens_re = []
for doc in corpus:
    tokens_re.append(regtok.tokenize(doc))

tokens_re

# strips out other bits

# finally, a tokenizer to help with twitter, and perhaps other social data
social = TweetTokenizer()

tokens_social = []
for doc in corpus:
    tokens_social.append(social.tokenize(doc))

tokens_social



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
combined = CountVectorizer(tokenizer=tokenize)
bow = combined.fit_transform(corpus)

bowdf = pd.DataFrame(bow.toarray(), columns=combined.get_feature_names())
bowdf


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

tfidf = TfidfVectorizer(token_pattern="[\w']+", ngram_range=(1,2))
tfidf.fit(corpus)

# lets put this into a dataframe
idf = tfidf.transform(corpus)

idf = pd.DataFrame(idf.toarray(), columns=tfidf.get_feature_names())


# we could even heatmap this to help understand the intuition here

plt.figure(figsize=(4,6))
sns.heatmap(idf.T, xticklabels=True, yticklabels=True, cmap='Reds')
plt.show()

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

corpus2 = corpus.copy()
corpus2.extend(["I want help", "halp", "python is fun", "Can I get help"])

# refit - but simplify, just tokens (unigrams)

tfidf2 = TfidfVectorizer(token_pattern="[\w']+", ngram_range=(1,1))
tfidf2.fit(corpus2)
idf2 = tfidf2.transform(corpus2)
idf2 = pd.DataFrame(idf2.toarray(), columns=tfidf2.get_feature_names())
plt.figure(figsize=(4,6))
sns.heatmap(idf2.T, xticklabels=True, yticklabels=True, cmap="Reds")
plt.show()

idf2.values

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

from sklearn.manifold import TSNE

tsne = TSNE()
t2 = tsne.fit_transform(idf2)

# we just compressed the tfidf array to 2d

# lets plot

# now lets fit a kmeans to the dataframe
# what falls out with k = 2?
# OVERLY simplistic of course, but the hardest part was getting the feature vectors setup via tokenization
# obviously we have some flexibility on how we tokenize, the ngrams, stopwords, max threshold, etc
# so solutions can take different forms!

from sklearn.cluster import KMeans
k2 = KMeans(2)
k2.fit(t2)
labs = k2.predict(t2)


word_cluster = pd.DataFrame(t2, columns=["x", "y"])
word_cluster
word_cluster['labs'] = labs

sns.scatterplot(x="x", y="y", data=word_cluster, hue="labs")
plt.show()


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
## questrom:datasets.topics (Big Query)
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
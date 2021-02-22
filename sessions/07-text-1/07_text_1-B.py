# LEARNING GOALS
#
#                 - text as a datasource
#                 - cleaning text
#                 - basic eda
#                 - Doc Term Matrix representation by hand
#                 - The intuition behind working with text before jumping into tools that abstract this away
#                 - how text can be used in ML

######################################## some helpful resources:
# https://www.w3schools.com/python/python_regex.asp
# https://docs.python.org/3/library/re.html
# https://www.debuggex.com/cheatsheet/regex/python
# https://www.shortcutfoo.com/app/dojos/python-regex/cheatsheet

# install
# ! pip install newspaper3k
# ! pip install spacy
# ! pip install wordcloud
# ! pip install emoji
# ! pip install nltk
# ! pip install scikit-plot

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

# a simple corpus
a = ['I like turtles!',
     'You like hockey and golf ',
     'Turtles and hockey ftw',
     'Python is very easy to learn. 🐍',
     'A great resource is www.spacy.io',
     ' Today is the Feb 22, 2021 !           ',
     '@username #hashtag https://www.text.com',
     'BA820 ']

df = pd.DataFrame({'text':a})
df

## QUICK QUESTION
##        What do you see about the data being brought in?

## we can always get the values back
df.text.values


# quick review of some of the string funcationality
# we saw in 765

# capitalize or change case
# upper, lower, strip
df.text.str.upper()
df.text.str.lower()
df['tokens'] = df.text.str.split()
df.tokens


# we can detect
df.text.str.contains('turtle')

# remember python is case sensitive!
df.text.str.contains('Turtle')

# we can replace anything that matches a pattern
# but we will come back to patterns
df.text.str.replace("a", "ZZZZZZZZZ")

# we can look at the length
df['len'] = df.text.str.len()
df

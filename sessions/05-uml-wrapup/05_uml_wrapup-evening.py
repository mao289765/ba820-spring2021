######################################################
## Wrap up UML
## Hands on-heavy class to dive into some concepts you may want to explore 
## Learning objectives:
##
## 0. finish up PCA
## 1. exposure to more contemporary techniques for interviews, awareness, and further exploration
## 2. highlight different use-cases, some that help with viz for non-tech, others to think about alternatives to linear PCA
## 3. use this as a jumping off point for you to consider the fact that are lots of methods, and its not typically for this task, do this one approach
## 4. hands on application of UML and SML data challenge!
##

# installs

# notebook install
# ! pip install umap-learn

# local
# pip install umap-learn

# imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

# scipy
from scipy.spatial.distance import pdist

# scikit
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.cluster import KMeans


######################################################
## ANOTHER APPROACH: tsne
## 
## preserve local structures (distance/neighborhoods) by taking high dimensional data and 
## representing that structure in a 2d space
## NON-LINEAR APPROACH
## 
## can be used as a way to visualize a highly dimensional dataset
## but we can think of the new dimensions as features/embeddings of the larger space
## 
##
## Let's see this interactively!
## https://distill.pub/2016/misread-tsne/
##
## NOTE: it can be slow for "larger" datasets, so be careful
## 
## one common pattern to help with (not solve) is to use PCA to get new features, and then apply TSNE
## these techniques highlight how we can put use all of the tools in our toolbox!
##
## Lets use tsne to show how UML and SML can come together!

# load the digits dataset

from sklearn.datasets import load_digits
digits = load_digits()

X = digits.data
y = digits.target
img = digits.images[0]

# extract the data for tsne
# we will use X for the "fit", y for profiling the output

# we extracted the data, but we also have images

# lets plot it up

plt.imshow(img, cmap="gray")
plt.title(f"Label: {y[0]}")
plt.show()

# do you remember I talked about flattened datasets for images
# 1 "row" per image, and the pixels as features?
img.flatten().shape
img.shape


# QUICK THOUGHT EXPERIMENT:
# we just saw how easy it is to take a 1 channel image and put it in a shape for machine learning@!
# however, what do you think the tradeoff might be as opposed to the natural 8x8 pixels

# lets fit a simple model to see how well we can classify the model before tsne

X.shape

from sklearn.tree import DecisionTreeClassifier  # simple decision tree
tree = DecisionTreeClassifier(max_depth=4)   # max depth of tree of 4 is random for example
tree.fit(X, y)  # sklearn syntax is everywhere!
tree_preds = tree.predict(X)   # 
tree_acc = tree.score(X, y)
tree_acc

# ^^^ this is our baseline

# 64 isn't that bad, but lets use PCA -> tsne which is generally the flow for "real" or "wide" datasets

# two step process - use PCA to reduce, then tsne for embeddings (2d)
# to keep things simple, just random choice for 90%, this could be 80%, or 95%

pca_m = PCA(.9)
pca_m.fit(X)
pcs_m = pca_m.transform(X)

pcs_m.shape

# # proof
np.sum(pca_m.explained_variance_ratio_)

# # how many components did we keep?
pca_m.n_components_


# # check this a different way
# pcs_m.shape

# Remember I said TSNE can be slow?
# step 2, tsne -- takes a few moments, even with the modest dataset that we are using

tsne = TSNE()
tsne.fit(pcs_m)

# get the embeddings
te = tsne.embedding_


#
# the shape
te.shape


# we know that one of the aims is that tsne helps re-map our data and find the structure
# 2d tsne

tdata = pd.DataFrame(te, columns=["e1", "e2"])
tdata['y'] = y

tdata.head(3)

tdata.y.value_counts(sort=False)

# the plot

PAL = sns.color_palette("bright", 10) 
plt.figure(figsize=(10, 8))
sns.scatterplot(x="e1", y="e2", hue="y", data=tdata, legend="full", palette=PAL)
plt.show()

# and finally, lets see if we improve the tree now?

X2 = tdata.drop(columns="y")
y2 = tdata.y

tree2 = DecisionTreeClassifier(max_depth=4)
tree2.fit(X2, y2)
tree2_preds = tree2.predict(X2)
tree2_acc2 = tree2.score(X2, y2)
tree2_acc2

# what was the first tree again?

################### YOUR TURN: TSNE Quick hands on practice!
## keep just the numeric columns from diamonds
## take a random sample of 1000 records
## exclude price!
## compress the features to 2d   <------- remember to be patient, tsne isn't "fast"
## visualize the embeddings with price as an overlay
## BONUS?  can you fit a linear regression in scikit to predict
##         price from the new 2d dimensions?

#####################################   TSNE Review
## 
## what did we just see?
## 
## we saw it took a few moments, thats usually the complaint
## one recommendation is to do PCA, but we even saw that it still wasn't immediate on a relatively small dataset
## however, even with the base settings, we got real good separation
## and a good jump in accuracy!
## 
## almost always 2 components (max 3, but I have only seen two in practice)
## 
## the parameters can be tweaked, and that is worth considering!
## perplexity: between 5 and 50, review the resource to see how this can arrive at different solutions
## number of iterations: docs suggest at least 250, 1k by default
##

############################## UMAP
## 
## another non-linear approach
## some research suggests that UMAP can better preserver the "relationships" 
## when going from high dimensional - lower dimensions
## 
## newer approach (2018)
## 
## some research suggests better fit (global structure retention), but its also much faster
## reduces the need for pre-processing steps like PCA
## can fit more dimensions than TSNE,
##
##
## https://pair-code.github.io/understanding-umap/
## 
## 
## params to consider
##  neighbors (most important)
##  random_state (to help with reproducibility)
##  number of components (can b2 more than 2, but remember the goal is we want to reduce our feature space!)
##  min_dist = review the tools in resources to see how different datasets form different solutions with changing parameters
##       - distance in low dimensional space
##       - smaller values tend to create tighltly compact embeddings, larger values are "looser" 
##  metric = not heavily discussed, but we can also use different distance 
##
##  ^^ Distance is everywhere in ML!

# load umap - violating my rule of thumb

from umap import UMAP

# rebuild the digits 

from sklearn.datasets import load_digits
digits = load_digits()

X = digits.data
y = digits.target

# docs, note the args/kwargs

# can start to type, will find params
u = UMAP(random_state=820, n_neighbors=10)
u.fit(X)

embeds = u.transform(X)

# type
type(embeds)

# shape
embeds.shape

# lets put this back into a dataframe
umap_df = pd.DataFrame(embeds, columns=["x", "y"])
umap_df['label'] = y




# what do we have for a scatterplot?  overlay the label 


PAL = sns.color_palette("bright", 10) 
plt.figure(figsize=(10, 8))
sns.scatterplot(x="x", y="y", hue="label", data=umap_df, legend="full", palette=PAL)
# sns.scatterplot(x="e1", y="e2", hue="y", data=tdata, legend="full", palette=PAL)
plt.show()

################################## Short(er) Breakout Challenge
## use diamonds dataset, take a sample of 5000 rows 
## keep just the numeric columns, all numeric including price
## use tsne and umap to generate 2 new dimensions
## create vis to compare
## create two kmeans = 5 cluster solutions (think: proxy for cut?)
## use tsne embeddings for cluster
## use umap for cluster
## which has a better silhouette score?
## overlay the cut column onto the clusters - profile each cluster by cut, is there agreement?
# cluster assignment agreement?  Simply, did the two approaches tend to find similar clusters?

################################### Breakout Challenge
## work as a group to combine UML and SML!
## housing-prices tables on Big Query will be used questrom.datasets._______
##     housing-train = your training set
##     housing-test = the test set, does not have the target, BUT does have an ID that you will need for your submission
##     housing-sample-submission = a sample of what a submission file should look like, note the id and your predicted value
## 
## use regression to predict median_house_value
## 
## you can use all of the techniques covered in the program, and this course
## objective:  MAE - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
##
##
## tips/tricks
##    - ITERATE!  iteration is natural, and your friend
##    - submit multiple times with the same team name
##    - what would you guess without a model, start there!
##    - you will need to submit for all IDs in the test file
##    - it will error on submission if you don't submit 
##
## Leaderboard and submissions here: http://34.86.144.106:8501/
##



################################################# Other approaches to consider!
##  MDS - addded as bonus to this session folder
## 
##  Factor Analysis
##     -a method to identify latent constructs
##     - Techniques that can be applied to market research, 
##        product management and consumer insights
##     - similar to PCA but we don't care about info explained, moreso on the proper construction
##       of the constructs that we use for summarization
##
##  LDA - Linear Discriminant Analysis
##     - Like PCA, but we give it a label to help guide the features during reduction
##     - https://machinelearningmastery.com/linear-discriminant-analysis-for-dimensionality-reduction-in-python/#:~:text=Linear%20Discriminant%20Analysis%2C%20or%20LDA,examples%20by%20their%20assigned%20class.
## 
##  Other variances of PCA even
##     - Randomized PCA
##     - Sparse PCA
##
##  Recommendation Engines
##      - extend "offline" association rules 
##      - added some links to the resources (great article with other libraries)
##      - toolkits exist to configure good approaches for real-use
##      - I call reco engines unsupervised because its moreso about using neighbors and similarity to back 
##        into items to recommend
##      - can be done by finding similar users, or similar items.
##      - hybrid approaches work too
##      - scikit surprise
##      NOTE:  Think about it? you can pull data from databases! you saw flask APIs!  Build your own reco tool!
##             batch calculate recos and store in a table, send user id to API, look up the previously made recommendations
##             post feedback to database, evaluate, iterate, repeat!
##     
##   A python package to review
##      - I followed this package in its early days (graphlab) before Apple bought the company
##      - expressive, with pandas-like syntax
##      - accessible toolkit for a number of ML tasks, including Reco engines 
##      - https://github.com/apple/turicreate




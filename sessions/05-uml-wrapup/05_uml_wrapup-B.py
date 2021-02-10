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

########################################################################
# PCA wrapup - quick individual exercise
# 
# Let's get coding!
# 
# 1. get the diamonds dataset from Big Query
# 2. keep just the numeric columns, but EXCLUDE price
# 3. standardize the columns with Min Max Scaler (tricky!)  <----- for practice purposes
# 4. fit a PCA model to the data
# 5. plot the variance explained (your choice)
#

SQL = "select * from `questrom.datasets.diamonds`"
dia = pd.read_gbq(SQL, "questrom")


dia_n = dia.select_dtypes("number")
X = dia_n.drop(columns="price")
y = dia_n.price

mm = MinMaxScaler()
mm.fit(X)
Xs = mm.transform(X)
pca = PCA()
pca.fit(Xs)
pcs = pca.transform(Xs)

# 0 = 1st component
plt.plot(range(pca.n_components_), pca.explained_variance_ratio_)
plt.show()

pcs.shape
Xs.shape

df = pd.DataFrame(pcs)
df.head(3)

df2 = df.iloc[:, :2]
df2.head(3)

np.cumsum(pca.explained_variance_ratio_)


#############################  MOVING FORWARD
## 
## some options when fitting PCA Model
## we can pre-specify the number of components OR the % of var we want to explain!
## lets use diamonds again but take the numeric columns and standard scale

# lets rebuild to practice and show some quick 1-liners
# just in case

dia_n = dia.select_dtypes("number")
dia_s = StandardScaler().fit_transform(dia_n)

# arbitrary selection, but lets keep just 2 components
# using svd_solver=full to keep solutions consistent
# scikit tries to be smart with auto, but I want to the same solver for examples
pc2 = PCA(2, svd_solver="full")
dia_pc2 = pc2.fit_transform(dia_s)


# variance explained in just 2?
np.cumsum(pc2.explained_variance_ratio_)

# I will violate my rule and put back onto the original just for exploration
# just to explore the fits
diamonds = dia.copy()
diamonds[['pc2_1', 'pc2_2']] = dia_pc2
diamonds.head(3)



# remember viz is a use case of dimensionality reduction?
# https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

# overlay the category
diamonds['cut2'] = diamonds.cut.astype('category').cat.codes
plt.scatter(x=diamonds.pc2_1, y=diamonds.pc2_2, c=diamonds.cut2, cmap=cm.Paired, alpha=.3)
plt.show()

# lets fit another PCA model, but keep 90% variance
pc90 = PCA(.9, svd_solver="full")
dia_pc90 = pc90.fit_transform(dia_s)
dia_pc90.shape

np.cumsum(pc90.explained_variance_ratio_)

# Violate again but putting back onto same dataset, but see results

diamonds[['pc90_1', 'pc90_2', 'pc90_3']] = dia_pc90

# now lets look at everything
diamonds.head(3)

###########################
## Summary
##
## goal is to reduce features and keep the core information
## we CAN fit the model to the full dataset
## if you want to subset, you can do up front, but we saw how to evaluate when we may not know what to select
##

##################################################################
##
##   PCA Compression
## 
## once we have fit our model, we have seen that transform genera†es the newly constructed features
## but we can also project back into the original feature space

# quick refresher, what was the shape that we used?
# scaled diamonds numeric without price

dia_s.shape
dia_pc90.shape

# lets use the fit where we specified 90% to fit

# we can use inverse transform to do this
comp3 = pc90.inverse_transform(dia_pc90)
comp3.shape

# put back into a dataframe
# remember, I scaled dia_n -> dia_s
comp3df = pd.DataFrame(comp3, columns=dia_n.columns)
comp3df.head(3)

# remember the MNIST dataset I showed on slides
# 
# I showed the aspect of compression there, a simple link
# the digit is reconstructed after fitting N components and .inverse_tranform
# depends on our use case of course, but highlights that "losing" information may not hurt your applications
# depending on the dataset, our problem, and our method, we can potentially remove noise and only retain signal!

# https://snipboard.io/qrohR7.jpg

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

# from sklearn.datasets import load_digits
# digits = load_digits()

# extract the data for tsne
# we will use X for the "fit", y for profiling the output

# we extracted the data, but we also have images

# lets plot it up

# plt.imshow(img, cmap="gray")
# plt.title(f"Label: {y[0]}")
# plt.show()

# do you remember I talked about flattened datasets for images
# 1 "row" per image, and the pixels as features?

# QUICK THOUGHT EXPERIMENT:
# we just saw how easy it is to take a 1 channel image and put it in a shape for machine learning@!
# however, what do you think the tradeoff might be as opposed to the natural 8x8 pixels

# lets fit a simple model to see how well we can classify the model before tsne

# from sklearn.tree import DecisionTreeClassifier  # simple decision tree
# tree = DecisionTreeClassifier(max_depth=4)   # max depth of tree of 4 is random for example
# tree.fit(X, y)  # sklearn syntax is everywhere!
# tree_preds = tree.predict(X)   # 
# tree_acc = tree.score(X, y)
# tree_acc

# ^^^ this is our baseline

# 64 isn't that bad, but lets use PCA -> tsne which is generally the flow for "real" or "wide" datasets

# two step process - use PCA to reduce, then tsne for embeddings (2d)
# to keep things simple, just random choice for 90%, this could be 80%, or 95%

# pca_m = PCA(.9)
# pca_m.fit(X)
# pcs_m = pca_m.transform(X)

# # proof
# np.sum(pca_m.explained_variance_ratio_)

# # how many components did we keep?
# pca_m.n_components_

# # check this a different way
# pcs_m.shape

# Remember I said TSNE can be slow?
# step 2, tsne -- takes a few moments, even with the modest dataset that we are using

# tsne = TSNE()
# tsne.fit(pcs_m)

# get the embeddings


#
# the shape

# we know that one of the aims is that tsne helps re-map our data and find the structure
# 2d tsne

# tdata = pd.DataFrame(te, columns=["e1", "e2"])
# tdata['y'] = y

# tdata.head(3)

# tdata.y.value_counts(sort=False)

# the plot

# PAL = sns.color_palette("bright", 10) 
# plt.figure(figsize=(10, 8))
# sns.scatterplot(x="e1", y="e2", hue="y", data=tdata, legend="full", palette=PAL)

# and finally, lets see if we improve the tree now?

# X2 = tdata.drop(columns="y")
# y2 = tdata.y

# tree2 = DecisionTreeClassifier(max_depth=4)
# tree2.fit(X2, y2)
# tree2_preds = tree2.predict(X2)
# tree2_acc2 = tree2.score(X2, y2)
# tree2_acc2

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

# from umap import UMAP

# rebuild the digits 

# from sklearn.datasets import load_digits
# digits = load_digits()

# X = digits.data
# y = digits.target

# docs, note the args/kwargs

# can start to type, will find params

# type

# shape

# lets put this back into a dataframe

# what do we have for a scatterplot?  overlay the label 
# sns.scatterplot(x="x", y="y", hue="label", data=umap_df)

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




# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 23:27:13 2019

@author: Poorvi Prakash
"""
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
import plotly as py
import plotly.graph_objs as go
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup

#importing the tsv file into dataframe
dataset = pd.read_table('DatasetOfFMTweets.txt', encoding = "ISO-8859-1")
corpus = []
#DatasetOfFMTweets
#cleaning up 
for i in range(0,12894):
        # removing html encoding
        tweet = BeautifulSoup(dataset['text'][i], 'lxml').getText() 
        #removing @ from tweets
        tweet = re.sub(r'@[A-Za-z0-9_]+',' ',tweet)
        #removing urls from tweets
        tweet = re.sub('https?://[A-Za-z0-9./]+','',tweet)
        #testing = dataset['text'][16].decode("utf-8-sig")
        #removing hashtags, numbers and special characters 
        tweet = re.sub("[^a-zA-Z]", " ",tweet)
        #Converting the tweet to lower cases
        tweet = tweet.lower()
        #splitting string to list
        tweet = tweet.split()        
        #initializing PorterStemmer object for stemming and removing stopwords.
        ps = PorterStemmer()
        tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
        #Joining the list to a string
        tweet = " ".join(tweet)
        corpus.append(tweet)
        
#Creating sparse matrix using Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 12800)
X = cv.fit_transform(corpus).toarray()
cv.get_feature_names()
dataset.loc[dataset["gender"]=='male',"gender"]=0
dataset.loc[dataset["gender"]=='female',"gender"]=1
y = dataset.iloc[:, 1].values
#Dataframe just to view the data
x = pd.DataFrame(corpus)
y=y.astype('int')
Y = pd.DataFrame(y)
#Combining Corpus and result data in one dataframe
df= pd.concat([x,Y], axis = 1)
#Renaming columns
df.columns= ['text','Gender']
tweets_all = list(df.shape)[0]
tweets_categories = df['Gender'].value_counts()
print("\n \t The data has {} tweets, {} male and {} female.".format(tweets_all,tweets_categories[0],tweets_categories[1]))

#Using Word Cloud to visualize the data
from wordcloud import WordCloud
#Visualing only those words frequently used by male
print("\n \t WordCloud of Male tweets")
male_tweets = df[df.Gender == 0]
male_string = []
for t in male_tweets.text:
    male_string.append(t)
male_string = pd.Series(male_string).str.cat(sep=' ')
wordcloud = WordCloud(width=1600, height=800,max_font_size=200,colormap='magma').generate(male_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

from wordcloud import WordCloud
#Visualing only those words frequently used by female
print("\n \t WordCloud of Female tweets")
female_tweets = df[df.Gender == 1]
female_string = []
for t in female_tweets.text:
    female_string.append(t)
female_string = pd.Series(female_string).str.cat(sep=' ')
wordcloud = WordCloud(width=1600, height=800,max_font_size=200,colormap='magma').generate(female_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

"""
#Visualing only those words frequently used by both genders
vis_tweets = []
for t in corpus:
    vis_tweets.append(t)
vis_tweets = pd.Series(vis_tweets).str.cat(sep=' ')
wordcloud = WordCloud(width=1600, height=800,max_font_size=200,colormap='magma').generate(vis_tweets) 
plt.figure(figsize=(12,12))
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis("off") 
plt.show()
"""
#Spliting data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


from sklearn.metrics import accuracy_score
import time
accuracy=[]
time_data = []
#Fitting naive bayes to training set

from sklearn.naive_bayes import GaussianNB
start = time.time()
classifier = GaussianNB()  
classifier.fit(X_train, y_train)  

#predicting for the test set
y_pred = classifier.predict(X_test)   

end = time.time()

#Making confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
accuracy.append(accuracy_score(y_pred, y_test))
time_data.append(end-start)
print("\n \tNaive Bayes Accuracy: {0:.2%}".format(accuracy_score(y_pred, y_test)))
print("\n \tNaive Bayes Execution time: {0:.5} seconds \n".format(end-start))

#Accuracy of Naive Bayes
#((1110+360)/2579)*100 = 56.9988367

#Fitting Decision Tree classifier tp training set
from sklearn.tree import DecisionTreeClassifier
start = time.time()
decisionClassifier = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
decisionClassifier.fit(X_train, y_train)

#predicting for test set
y_predDecision = decisionClassifier.predict(X_test)
Y_predDecision = pd.DataFrame(y_predDecision)
end = time.time()

#Making confusion Matrix
from sklearn.metrics import confusion_matrix
cmd = confusion_matrix(y_test, y_predDecision)
accuracy.append(accuracy_score(y_predDecision, y_test))
time_data.append(end-start)
print("\n \tDecision Tree Accuracy: {0:.2%}".format(accuracy_score(y_predDecision, y_test)))
print("\n \tDecision Tree Execution time: {0:.5} seconds \n".format(end-start))


#Accuracy of Decision Tree
#((805+619)/2579)*100 = 55.2151996


#Fitting Random Forest classifier to training set
from sklearn.ensemble import RandomForestClassifier
start = time.time()
randomClassifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
randomClassifier.fit(X_train, y_train)

#predicting for test set
y_predRandom = randomClassifier.predict(X_test)
Y_predRandom = pd.DataFrame(y_predRandom)
end = time.time()

#Making confusion Matrix
from sklearn.metrics import confusion_matrix
cmr = confusion_matrix(y_test, y_predRandom)
accuracy.append(accuracy_score(y_predRandom, y_test))
time_data.append(end-start)
print("\n \tRandom Forest Accuracy: {0:.2%}".format(accuracy_score(y_predRandom, y_test)))
print("\n \tRandom Forest Execution time: {0:.5} seconds \n".format(end-start))

#Accuracy of Random Forest Tree
#((962+610)/2579)*100 = 60.9538580

#for large datasets no: of trees can be up to 200


#Fitting Logistic Regression classifier to training set
from sklearn.linear_model import LogisticRegression
start = time.time()
logisticClassifier = LogisticRegression(random_state = 0)
logisticClassifier.fit(X_train, y_train)

#predicting for test set
y_predLR = logisticClassifier.predict(X_test)
Y_predLR = pd.DataFrame(y_predLR)
end = time.time()

#Making confusion Matrix
from sklearn.metrics import confusion_matrix
cmlr = confusion_matrix(y_test, y_predLR)
accuracy.append(accuracy_score(y_predLR, y_test))
time_data.append(end-start)
print("\n \tLogistic Regression Accuracy: {0:.2%}".format(accuracy_score(y_predLR, y_test)))
print("\n \tLogistic Regression Execution time: {0:.5} seconds \n".format(end-start))
#Accuracy of Logistic Regression 
#((845+672)/2579)*100 = 58.8212485

#Fitting KNN classifier to training set
from sklearn.neighbors import KNeighborsClassifier
start = time.time()
KNNClassifier = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski',p=2)
KNNClassifier.fit(X_train, y_train)

#predicting for test set
y_predKNN = KNNClassifier.predict(X_test)
Y_predKNN = pd.DataFrame(y_predKNN)
end = time.time()

#Making confusion Matrix
from sklearn.metrics import confusion_matrix
cmknn = confusion_matrix(y_test, y_predKNN)
accuracy.append(accuracy_score(y_predKNN, y_test))
time_data.append(end-start)
print("\n \tKNN Accuracy: {0:.2%}".format(accuracy_score(y_predKNN, y_test)))
print("\n \tKNN Execution time: {0:.5} seconds \n".format(end-start))

#Accuracy of KNN 
#((1076+249)/2579)*100 = 51.3765025

#Fitting svm classifier to training set
from sklearn.svm import SVC
start = time.time()
SVMClassifier = SVC(kernel = 'linear', random_state = 0)
SVMClassifier.fit(X_train, y_train)

#predicting for test set
y_predsvm = SVMClassifier.predict(X_test)
Y_predsvm = pd.DataFrame(y_predsvm)
end = time.time()

#Making confusion Matrix
from sklearn.metrics import confusion_matrix
cmsvm = confusion_matrix(y_test, y_predsvm)
accuracy.append(accuracy_score(y_predsvm, y_test))
time_data.append(end-start)
print("\n \tSVM Accuracy: {0:.2%}".format(accuracy_score(y_predsvm, y_test)))
print("\n \tSVM Execution time: {0:.5} seconds \n".format(end-start))

#Accuracy of svm 
#((826+640)/2579)*100 = 56.8437378

#Fitting kernel svm to training set
from sklearn.svm import SVC
start = time.time()
ksvmClassifier = SVC(kernel = 'rbf', random_state=0 )
ksvmClassifier.fit(X_train, y_train)  

#predicting for the test set
y_predKSVM = ksvmClassifier.predict(X_test) 
Y_predKsvm = pd.DataFrame(y_predKSVM)  
end = time.time()

#Making confusion Matrix
from sklearn.metrics import confusion_matrix
ksvmcm = confusion_matrix(y_test, y_predKSVM)
accuracy.append(accuracy_score(y_predKSVM, y_test))
time_data.append(end-start)
print("\n \tKSVM Accuracy: {0:.2%}".format(accuracy_score(y_predKSVM, y_test)))
print("\n \tKSVM Execution time: {0:.5} seconds \n".format(end-start))

#Accuracy of ksvm 
#((1331)/2579)*100 = 51.6091508

from plotly.offline import *
data = [go.Bar(
            x=['Naive Bayes', 'Decision Tress', 'Random Forest', 'Logistic Regression', 'KNN', 'SVM', 'KSVM'],
            y=[accuracy[0]*100, accuracy[1]*100, accuracy[2]*100, accuracy[3]*100,accuracy[4]*100, accuracy[5]*100, accuracy[6]*100]    
    )]
py.offline.plot(data, filename='comparison-bar.html')


from plotly.offline import *
data_time = [go.Bar(
            x=['Naive Bayes', 'Decision Tress', 'Random Forest', 'Logistic Regression', 'KNN', 'SVM', 'KSVM'],
            y=[time_data[0]*100, time_data[1]*100, time_data[2]*100, time_data[3]*100,time_data[4]*100, time_data[5]*100, time_data[6]*100]    
    )]
py.offline.plot(data_time, filename='comparison-time-bar.html')



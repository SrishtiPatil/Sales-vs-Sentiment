# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 18:59:27 2020

@author: Srishti
"""


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
from wordcloud import WordCloud 
import matplotlib.pyplot as plt

file = './Polarity.xlsx'
xls = pd.ExcelFile(file)
df = xls.parse('Sheet1')
fig, (ax2) = plt.subplots(1,1,figsize=[17, 10])
wordcloud2 = WordCloud(background_color='black',colormap="Blues", 
                        width=600,height=400).generate(" ".join(df['text_token']))

ax2.imshow(wordcloud2,interpolation='bilinear')
ax2.axis('off')
ax2.set_title('Most Used Words in Comments',fontsize=35)
#Linear Regression
file = './Polarity.xlsx'
xls = pd.ExcelFile(file)
df = xls.parse('NumericPolarity')
df
df.columns

from sklearn.linear_model import LinearRegression
X_train, X_test, Y_train, Y_test = train_test_split(df['text_token'], df['Polarity'], test_size=0.2, random_state=0)


vectorizer = TfidfVectorizer(min_df=10)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
lr = LinearRegression().fit(X_train, Y_train)



print("Training set score: {:.3f}".format(lr.score(X_train, Y_train)))
print("Test set score: {:.3f}".format(lr.score(X_test, Y_test)))

lrt1=lr.score(X_train, Y_train)
lrt2=lr.score(X_test, Y_test)

#Logistic Regression
file = './Polarity.xlsx'
xls = pd.ExcelFile(file)
df = xls.parse('Sentiment')


X_train, X_test, Y_train, Y_test = train_test_split(df['text_token'], df['Polarity_Category'], test_size=0.2, random_state=0)

vectorizer = TfidfVectorizer(min_df=10)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


logreg = LogisticRegression(penalty="l2", C=10)
logreg.fit(X_train, Y_train)


print("Training set score: {:.3f}".format(logreg.score(X_train, Y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, Y_test)))
logt1=logreg.score(X_train, Y_train)
logt2=logreg.score(X_test, Y_test)



#MultiNomial Naive Baysian - this is used when text data needs to be classified basis probability
X_train, X_test, Y_train, Y_test = train_test_split(df['text_token'], df['Polarity_Category'], test_size=0.2, random_state=0)


vectorizer = TfidfVectorizer(min_df=10)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

mnb = MultinomialNB()

mnb.fit(X_train,Y_train)



predicted = mnb.predict(X_test)


print("Training set score: {:.3f}".format(mnb.score(X_train, Y_train)))
print("Test set score: {:.3f}".format(mnb.score(X_test, Y_test)))

mnbt1=mnb.score(X_train, Y_train)
mnbt2=mnb.score(X_test, Y_test)


#SVC
X_train, X_test, Y_train, Y_test = train_test_split(df['text_token'], df['Polarity_Category'], test_size=0.2, random_state=0)

vectorizer = TfidfVectorizer(min_df=10)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

linsvc = LinearSVC()
linsvc.fit(X_train, Y_train)


print("Training set score: {:.3f}".format(linsvc.score(X_train, Y_train)))
print("Test set score: {:.3f}".format(linsvc.score(X_test, Y_test)))

linsvct1=linsvc.score(X_train, Y_train)
linsvct2=linsvc.score(X_test, Y_test)


#SVM OVO
from sklearn import svm
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X_train, Y_train)

print("Training set score: {:.3f}".format(clf.score(X_train, Y_train)))
print("Test set score: {:.3f}".format(clf.score(X_test, Y_test)))


#SVM OVR
clf = svm.SVC(decision_function_shape='ovr')
clf.fit(X_train, Y_train)

print("Training set score: {:.3f}".format(clf.score(X_train, Y_train)))
print("Test set score: {:.3f}".format(clf.score(X_test, Y_test)))


#GridSeach CV


from sklearn import svm
from sklearn.model_selection import GridSearchCV
file = './Polarity.xlsx'
xls = pd.ExcelFile(file)
df = xls.parse('Sentiment')

X = vectorizer.fit_transform(df['text_token'])
parameters = {'kernel':['rbf'], 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)

clf.fit(X,df['Polarity_Category'])
sorted(clf.cv_results_.keys())


print("Linear Reg Train: {:.3f}".format(lrt1))
print("Linear Reg Test: {:.3f}".format(lrt2))
print("Log Reg Train: {:.3f}".format(logt1))
print("Log Reg Test: {:.3f}".format(logt2))
print("Naive Baysian Train: {:.3f}".format(mnbt1))
print("Naive Baysian Test: {:.3f}".format(mnbt2))
print("Lin SVC Train: {:.3f}".format(linsvct1))
print("Lin SVC Test: {:.3f}".format(linsvct2))




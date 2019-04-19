# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import matplotlib.pyplot as plt

df_raw = pd.read_csv("Tweets.csv")
df = pd.read_csv("Tweets.csv")
df.head()
df.columns
df.airline.describe()
df.airline.value_counts()

df.isnull().sum()/len(df)
print(df.duplicated().sum())
print(len(df))
df.drop_duplicates(inplace = True)
len(df)

df = df.loc[df.airline_sentiment_confidence >= 0.5]

stop_words = set(stopwords.words('english'))
whitelist = ["n't", "not", "no"]
wordnet_lemmatizer = WordNetLemmatizer()

def normalizer(tweet):
    no_at = re.sub(r'@\w+', '', tweet)
    no_url = re.sub(r'http.?://[^\s]+[\s]?', ' ', no_at)
    only_letters = re.sub("[^a-zA-Z]", " ",no_url)
    tokens = nltk.word_tokenize(only_letters)
    lower_case = [l.lower() for l in tokens]
    filtered_result = [word for word in lower_case if (word not in stop_words or word in whitelist) and len(word) > 1] 
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    meaningful_words = [w for w in lemmas] 
    return( " ".join( meaningful_words )) 
    
#dftest = df.loc[:5].copy()
#dftest['clean_tweet']=dftest['text'].apply(lambda x: normalizer(x))
#dftest.clean_tweet

df['clean_tweet']=df['text'].apply(lambda x: normalizer(x))
df['tweet_len'] = df['clean_tweet'].apply(lambda x: len(x))

#df.loc[:,'sentiment'] = df.airline_sentiment.map({'negative':0,'neutral':2,'positive':4})
df.loc[(df.airline_sentiment=='positive'), 'sentiment']= 1
df.loc[(df.airline_sentiment=='neutral'), 'sentiment']= 0
df.loc[(df.airline_sentiment=='negative'), 'sentiment']= -1


X = df['clean_tweet']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)
'''
train_clean_tweet=[]
for tweet in X_train['clean_tweet']:
    train_clean_tweet.append(tweet)
test_clean_tweet=[]
for tweet in X_test['clean_tweet']:
    test_clean_tweet.append(tweet)'''
    
# Initialize a TfidfVectorizer object: tfidf_vectorizer
tfidf_vectorizer = TfidfVectorizer(analyzer = 'word', max_df=0.8)
# Transform the training data: tfidf_train 
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
#print(type(tfidf_train))
#print(tfidf_train.shape)
# Transform the test data: tfidf_test 
tfidf_test = tfidf_vectorizer.transform(X_test)
# Create a Multinomial Naive Bayes classifier: nb_classifier

nb_classifier = MultinomialNB(alpha = 0.1)
# Fit the classifier o the training data
#tfidf_train =pd.concat([pd.DataFrame(tfidf_train.todense()),df.tweet_len],axis=1)

nb_classifier.fit(tfidf_train, y_train)
# Create the predicted tags: pred
nbt_predt = nb_classifier.predict(tfidf_train)
nbt_pred = nb_classifier.predict(tfidf_test)
print("Training accuracy : {:.2%}".format(accuracy_score(nbt_predt, y_train)))
print("Testing accuracy : {:.2%}".format(accuracy_score(nbt_pred, y_test)))
print("Confusion Matrix:")
print(confusion_matrix(y_test,nbt_pred))
print("Classification Report:")
print(classification_report(y_test, nbt_pred))


# Perform 3-fold CV
cvscores5 = cross_val_score(nb_classifier, tfidf_train, y_train, cv = 5)

# Create the list of alphas: alphas
alphas = np.arange(0,1,0.1)

# Define train_and_predict()
def train_and_predict(alpha):
    # Instantiate the classifier: nb_classifier
    nb_classifier = MultinomialNB(alpha=alpha)
    # Fit to the training data
    nb_classifier.fit(tfidf_train, y_train)
    # Predict the labels: pred
    pred = nb_classifier.predict(tfidf_test)
    # Compute accuracy: score
    score = accuracy_score(y_test,pred)
    return score

# Iterate over the alphas and print the corresponding score
for alpha in alphas:
    print('Alpha: ', alpha)
    print('Score: ', train_and_predict(alpha))
    print()

#alpha = 0.1 Score:  0.7650273224043715

# Get the class labels: class_labels
class_labels = nb_classifier.classes_
# Extract the features: feature_names
feature_names = tfidf_vectorizer.get_feature_names()
# Zip the feature names together with the coefficient array and sort by weights: feat_with_weights
feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))
# Print the first class label and the top 20 feat_with_weights entries
print(class_labels[0], feat_with_weights[:20])
# Print the second class label and the bottom 20 feat_with_weights entries
print(class_labels[1], feat_with_weights[0:20]) 

print(df.tweet_created.max())
print(df.tweet_created.min())




svc = SVC(kernel="rbf", C=0.025, probability=True)
svc.fit(tfidf_train, y_train.values.ravel())
# Predict the labels: pred
svcpred = svc.predict(tfidf_test)
# Compute accuracy: score
score = accuracy_score(y_test,svcpred)
print(score)
print(classification_report(y_test,svcpred))



'''
from nltk.probability import FreqDist
fdist1 = FreqDist(X)
fdist1.most_common(50)

#df.to_csv('cleaned_tweets.csv')
df.columns
word_all =[]
for tw in df.clean_tweet:
    word_all.append(tw)
    
thefile = open('test.txt', 'w')
for item in word_all:
  thefile.write("%s\n" % item)   
'''
count_vectorizer = CountVectorizer()
# Transform the training data using only the 'text' column values: count_train 
count_train = count_vectorizer.fit_transform(X_train)
# Transform the test data using only the 'text' column values: count_test 
count_test = count_vectorizer.transform(X_test)
nb_classifier.fit(count_train, y_train)
# Create the predicted tags: pred
# Calculate the accuracy score: score
nbt_predtc = nb_classifier.predict(count_train)
nbt_predc = nb_classifier.predict(count_test)
#print("               ===== Naive Bayes =====")
print("Training accuracy : {:.2%}".format(accuracy_score(nbt_predtc, y_train)))
print("Testing accuracy : {:.2%}".format(accuracy_score(nbt_predc, y_test)))
print("Confusion Matrix:")
print(confusion_matrix(y_test,nbt_predc))
print("Classification Report:")
print(classification_report(y_test, nbt_predc))

cvscores_c5 = cross_val_score(nb_classifier, count_train, y_train, cv = 5)
fold = [1,2,3,4,5]

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(fold, cvscores5)
plt.xlim((1,5))
plt.ylim((0,1))
plt.xlabel('fold')
plt.ylabel('accuarcy')
plt.show()

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(fold, cvscores_c5)
plt.xlim((1,5))
plt.ylim((0,1))
plt.xlabel('fold')
plt.ylabel('accuarcy')
plt.show()

from matplotlib.ticker import MaxNLocator
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(fold, cvscores_r5)
#plt.xlim((1,5))
plt.axis([1, 5, 0, 1])
plt.xlabel('fold')
plt.ylabel('accuarcy')
plt.show()



rf = RandomForestClassifier(n_estimators=200)
rf.fit(tfidf_train, y_train)
rf_predt = rf.predict(tfidf_train)
rf_pred = rf.predict(tfidf_test)
print("Training accuracy : {:.2%}".format(accuracy_score(rf_predt, y_train)))
print("Testing accuracy : {:.2%}".format(accuracy_score(rf_pred, y_test)))
print("Confusion Matrix:")
print(confusion_matrix(y_test,rf_pred))
print("Classification Report:")
print(classification_report(y_test, rf_pred))

cvscores_r5 = cross_val_score(rf, tfidf_train, y_train, cv = 5)

xgb = XGBClassifier(max_depth=3, 
                    learning_rate=0.1, 
                    n_estimators=100, silent=True, 
                    objective='multi:softmax',
                    num_class = 3,
                    booster='gbtree', 
                    n_jobs=1)

xgb.fit(tfidf_train, y_train)
xgb_predt = xgb.predict(tfidf_test)

print("Training accuracy : {:.2%}".format(accuracy_score(xgb.predict(tfidf_train), y_train)))
print("Testing accuracy : {:.2%}".format(accuracy_score(xgb_predt, y_test)))
print("Confusion Matrix:")
print(confusion_matrix(y_test,xgb_predt))
print("Classification Report:")
print(classification_report(y_test, xgb_predt))

cvscores_xgbt = cross_val_score(xgb, tfidf_train, y_train, cv = 5)
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(fold, cvscores_xgbt)
#plt.xlim((1,5))
plt.axis([1, 5, 0, 1])
plt.xlabel('fold')
plt.ylabel('accuarcy')
plt.show()

xgb.fit(count_train, y_train)
xgb_pred = xgb.predict(count_test)

print("Training accuracy : {:.2%}".format(accuracy_score(xgb.predict(count_train), y_train)))
print("Testing accuracy : {:.2%}".format(accuracy_score(xgb_pred, y_test)))
print("Confusion Matrix:")
print(confusion_matrix(y_test,xgb_pred))
print("Classification Report:")
print(classification_report(y_test, xgb_pred)
#???
#lr = LogisticRegression(C=1,solver='liblinear',multi_class=”multinomial”, max_iter=200)

lr.fit(tfidf_train, y_train)
lr_pred = xgb.predict(tfidf_test)

print("Training accuracy : {:.2%}".format(accuracy_score(lr.predict(tfidf_train), y_train)))
print("Testing accuracy : {:.2%}".format(accuracy_score(lr_pred, y_test)))
print("Confusion Matrix:")
print(confusion_matrix(y_test,lr_pred))
print("Classification Report:")
print(classification_report(y_test, lr_pred))

cvscores_xgbt = cross_val_score(lr, tfidf_train, y_train, cv = 5)
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(fold, cvscores_xgbt)
#plt.xlim((1,5))
plt.axis([1, 5, 0, 1])
plt.xlabel('fold')
plt.ylabel('accuarcy')
plt.show()

word_all =[]
for tw in df.text:
    word_all.append(tw)
#word_all = word_all.decode('utf-8')    
thefile = open('text.txt', 'w',encoding='utf-8')
for item in word_all:
  thefile.write("%s\n" % item) 
  
'''import pip

pip.pep425tags.get_supported()'''


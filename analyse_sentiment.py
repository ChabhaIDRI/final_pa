#:\Users\MacBook-Pro-de-Chabha\AppData\Programs\Python\

# import de librairie 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# charger les données -> fichier csv qui contient des tweets et labels

tweets_df = pd.read_csv('twitter.csv')

# informations sur notre df, 3 colonnes : id, tweet, label

tweets_df.info()
tweets_df.describe()

## positive = tweets_df[tweets_df['label']==0]
## negative = tweets_df[tweets_df['label']==1]

#!pip install WordCloud

# afficher la ponctuation

import string
string.punctuation

exemple ='hi, how are you ?'
remove_punction = [char for char in exemple if char not in string.punctuation]
remove_punction_join = ''.join(remove_punction)


# charger les stopwords

import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words('english')

collect_stopwords=set(stopwords.words('english'))

remove_punction_stopwords = [txt for txt in remove_punction_join.split() if txt.lower() not in stopwords.words('english')]


def text_clean(text):
    remove_punction = [char for char in text if char not in string.punctuation]
    remove_punction_join = ''.join(remove_punction)
    remove_punction_stopwords = [txt for txt in remove_punction_join.split() if txt.lower() not in stopwords.words('english')]
    return remove_punction_stopwords



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tweets_df['tweet'], tweets_df['label'], test_size=0.2)


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, encoding='utf-8',
 decode_error='ignore')

vv = vectorizer.fit(X_train)
X_train=vectorizer.transform(X_train)
X_test=vectorizer.transform(X_test)



X = vv
y = tweets_df['label']


"""""
# transformer les données textuelles en vecteurs numériques 

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()


countvectorize_tweet = CountVectorizer(analyzer = text_clean, dtype = 'uint8').fit_transform(tweets_df['tweet']).toarray()

X = countvectorize_tweet
y = tweets_df['label']
X.shape
y.shape



# split les données en données d'entrainement et données de test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(countvectorize_tweet, tweets_df['label'], test_size=0.2)

"""
# utiliser le classifieur naive bayes

from sklearn.naive_bayes import MultinomialNB

NB_classify = MultinomialNB()
NB_classify.fit(X_train, y_train)

# le score du modèle

print("Le score du train : "+str(NB_classify.score(X_train,y_train)))
print("Le score du test : "+str(NB_classify.score(X_test,y_test)))

# récupérer le fichier du modele entrainé et celui du victor 

import warnings
import pickle
warnings.filterwarnings("ignore")

pickle.dump(NB_classify,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

pickle.dump(vectorizer,open('victor.pkl','wb'))
victor=pickle.load(open('victor.pkl','rb'))


pickle.dump(collect_stopwords,open('stopwords.pkl','wb'))
stopwords=pickle.load(open('stopwords.pkl','rb'))
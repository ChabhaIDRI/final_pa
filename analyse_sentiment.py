#:\Users\MacBook-Pro-de-Chabha\AppData\Programs\Python\

# import de librairie 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# charger les données -> fichier csv qui contient des tweets et labels

tweets_df = pd.read_csv('data.csv')
tweets_df.head(3)

# informations sur notre df, 3 colonnes : id, tweet, label

tweets_df.info()
tweets_df.describe()

## p = tweets_df[tweets_df['label']==0]
## n = tweets_df[tweets_df['label']==1]

# nettoyer les données

# supprimer la ponctuation

import string

# afficher la ponctuation !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
string.punctuation

exemple ='hi, how are you ?'

# split la phrase en char, on remplace 
# la ponctuation par des espaces
 
supprimer_ponctuation = [char for char in exemple if char not in string.punctuation]
supprimer_ponctuation_join = ''.join(supprimer_ponctuation)


# charger les stopwords : les mots en communs, qu'on utilise beaucoup

import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words('english')

collect_stopwords=set(stopwords.words('english'))

supprimer_ponctuation_stopwords = [txt for txt in supprimer_ponctuation_join.split() if txt.lower() not in stopwords.words('english')]


def text_clean(text):
    supprimer_ponctuation = [char for char in text if char not in string.punctuation]
    supprimer_ponctuation_join = ''.join(supprimer_ponctuation)
    supprimer_ponctuation_stopwords = [txt for txt in supprimer_ponctuation_join.split() if txt.lower() not in stopwords.words('english')]
    return supprimer_ponctuation_stopwords


# split les données

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tweets_df['tweet'], tweets_df['label'], test_size=0.2)

#tweets_df ['tweet'] = tweets_df ['tweet'].apply(text_clean)
#X_train, X_test, y_train, y_test = train_test_split(tweets_df ['tweet'], tweets_df['label'], test_size=0.2)




# Frequency Inverse Document Frequency
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, encoding='utf-8',
 decode_error='ignore')

vv = vectorizer.fit(X_train)
X_train=vectorizer.transform(X_train)
X_test=vectorizer.transform(X_test)

X = vv
y = tweets_df['label']


# utiliser le modele de la regression logistique

from sklearn.linear_model import LogisticRegression
model_lr=LogisticRegression(solver='liblinear')
model_lr.fit(X_train,y_train)

print("Score on training data is: "+str(model_lr.score(X_train,y_train)))
print("Score on testing data is: "+str(model_lr.score(X_test,y_test)))

# matrice de confusion

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

test = model_lr.predict(X_test)
matrice_c = confusion_matrix(y_test, test)
sns.heatmap(matrice_c, annot=True)

# récupérer le fichier du modele entrainé et celui du victor 

import warnings
import pickle
warnings.filterwarnings("ignore")

pickle.dump(model_lr,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

pickle.dump(vectorizer,open('victor.pkl','wb'))
victor=pickle.load(open('victor.pkl','rb'))


pickle.dump(collect_stopwords,open('stopwords.pkl','wb'))
stopwords=pickle.load(open('stopwords.pkl','rb'))

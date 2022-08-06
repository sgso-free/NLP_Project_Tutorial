import pandas as pd
import numpy as np
import re
import pickle

from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


#load data
url = "https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv"
df_raw = pd.read_csv(url)

#drop duplicates
df_raw = df_raw.drop_duplicates().reset_index(drop = True)

####PROCESS


df_interim = df_raw.copy()

def clean_data(urlData):
  
    #remove punctuation, digit, simbols
    urlData = re.sub('[^a-zA-Z]', ' ', urlData)
    
    #duplicate space
    urlData = re.sub(r'\s+', ' ',  urlData)
    #urlData=" ".join(urlData.split())

    urlData = re.sub(r'\b[a-zA-Z]\b', ' ',urlData)  #\b word boundary

    urlData = urlData.strip()   #remove space on right and left include tab
    return urlData


df_raw['url'] = df_raw['url'].str.lower() 
#clean-data
df_interim['url'] = df_interim['url'].apply(clean_data)

#fuction to reove stopwords
stopWord = ['is','you','your','and', 'the', 'to', 'from', 'or', 'I', 'for', 'do', 'get', 'not', 'here', 'in', 'im', 'have', 'on',
're', 'https', 'com', 'of']  

def remove_stopwords(urlData):
  if urlData is not None:
    words = urlData.strip().split()
    words_filtered = []
    for word in words:
      if word not in stopWord:
        words_filtered.append(word)
    result = " ".join(words_filtered) #hace un join elemento por elemento separados por espacio
  else:
      result = None
  return result

df_interim['url'] = df_interim['url'].apply(remove_stopwords)

#### MODEL

df = df_interim.copy()

X = df['url']
y = df['is_spam'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state = 42)

#Vectorizador
vec = CountVectorizer()

#create matrix
X_train = vec.fit_transform(X_train).toarray()
X_test = vec.transform(X_test).toarray()

#create the model using SVC
svclassifier = SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
svclassifier.fit(X_train, y_train)

#save the model to file
filename = 'models/svc_model.sav' #use absolute path
pickle.dump(modelo, open(filename, 'wb'))
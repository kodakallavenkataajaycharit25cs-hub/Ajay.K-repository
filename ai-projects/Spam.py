import numpy as np
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score
import gensim.models.word2vec

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
try:
  stopwords.words('english')
except LookupError:
  nltk.download('stopwords')
try:
  nltk.data.find('corpora/wordnet.zip')
except LookupError:
  nltk.download("wordnet")
def preprocess_text(text):
  text = text.lower()
  text = "".join([char for char in text if char not in string.punctuation])
  tokens = text.split()
  stop_words = set(stopwords.words('english'))
  tokens = [word for word in tokens if word not in stop_words]
  return ' '.join(tokens)  

script_dir = pathlib.Path(__file__).parent.resolve()
csv_path = script_dir / 'SMSSpamCollection.csv'
df = pd.read_csv(csv_path,sep= "\t",header = None,names=["label","message"],encoding='latin-1')
df['label'] = df["label"].map({"ham" : 0 , "spam": 1})
df['message'] = df["message"].apply(preprocess_text)

X = df["message"] 
y = df["label"]
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size= 0.2,random_state=42)

tf = TfidfVectorizer()
tf_train = tf.fit_transform(X_train)
tf_test = tf.transform(X_test)
model = MultinomialNB()
model.fit(tf_train,y_train)

y_pred_train = model.predict(tf_train)
train_accuracy = accuracy_score(y_train, y_pred_train) 
train_preci = precision_score(y_train,y_pred_train)
train_recall = recall_score(y_train,y_pred_train)
train_f1 = f1_score(y_train,y_pred_train)
print(f"Train Accuracy : {train_accuracy:.3f}")
print("Training Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))
print("Training Precision Score:\n", train_preci)
print("Training Recall Score:\n", train_recall)
print("Training F1-Score:\n", train_f1)

y_pred_test = model.predict(tf_test)
test_acc = accuracy_score(y_test, y_pred_test)
test_preci = precision_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)
print(f"Test Accuracy : {test_acc:.3f}")
print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
print("Test Precision Score:\n", test_preci)
print("Test Recall Score:\n", test_recall)
print("Test F1-Score:\n", test_f1)

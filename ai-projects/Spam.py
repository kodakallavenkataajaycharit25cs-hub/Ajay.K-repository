import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix

df = pd.read_csv("SMSSpamCollection.csv",sep= "\t",header = None,names=["label","message"],encoding='latin-1')
df['label'] = df["label"].map({"ham" : 0 , "spam": 1})

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
print(f"Train Accuracy : {train_accuracy:.3f}")
print("Training Confusion Matrix:\n", confusion_matrix(y_train, y_pred_train))

y_pred_test = model.predict(tf_test)
test_acc = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy : {test_acc:.3f}")
print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
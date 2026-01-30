import numpy as np
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score
import gensim.models.word2vec as Word2Vec
import torch
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import torch.nn as nn
import torch.optim as optim


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

def tokenize_text(text):
  text = text.lower()
  text = "".join([char for char in text if char not in string.punctuation])
  tokens = text.split()
  stop_words = set(stopwords.words('english'))
  tokens = [word for word in tokens if word not in stop_words]
  return tokens
def get_avg_word2vec(tokens,model,vector_size):
  word_vect = []
  for token in tokens:
    if token in model.wv:
      word_vect.append(model.wv[token])
  if word_vect:
    return np.mean(word_vect , axis=0)
  else:
    return np.zeros(vector_size)


script_dir = pathlib.Path(__file__).parent.resolve()
csv_path = script_dir / 'SMSSpamCollection.csv'
df = pd.read_csv(csv_path,sep= "\t",header = None,names=["label","message"],encoding='latin-1')
df['label'] = df["label"].map({"ham" : 0 , "spam": 1})
df['message'] = df["message"].apply(preprocess_text)

X = df["message"] 
y = df["label"]
X_train_raw , X_test_raw , y_train , y_test = train_test_split(X,y,test_size= 0.2,random_state=42)

X_train_tokens = X_train_raw.apply(tokenize_text)
X_test_tokens = X_test_raw.apply(tokenize_text)

w2v_model = Word2Vec.Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
w2v_model.build_vocab(X_train_tokens)
w2v_model.train(X_train_tokens, total_examples=w2v_model.corpus_count, epochs=10)
X_train_word2vec = np.array([get_avg_word2vec(tokens, w2v_model, 100) for tokens in X_train_tokens])
X_test_word2vec = np.array([get_avg_word2vec(tokens, w2v_model, 100) for tokens in X_test_tokens])

X_train = torch.from_numpy(X_train_word2vec).float()
X_test = torch.from_numpy(X_test_word2vec).float()
y_train = torch.from_numpy(y_train.values).long()
y_test = torch.from_numpy(y_test.values).long()

dropout = 0.2
model = nn.Sequential(nn.Linear(100,64),
      nn.ReLU(),
      nn.Dropout(p=dropout),
      
      nn.Linear(64,32),
      nn.ReLU(), 
      nn.Dropout(p=dropout),
      
      nn.Linear(32,16),
      nn.ReLU(),
      nn.Dropout(p=dropout),
      
      nn.Linear(16,2)
    )
#print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#print("Loss Funtion : ",criterion)
#print("Optimizer : ",optimizer)

epochs = 10000
for epoch in range(epochs):
  model.train()
  outputs = model(X_train)
  loss = criterion(outputs, y_train)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
  if (epoch+1) % 2000 == 0:
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
print("\nTraining finished!")

print("\n--- Evaluating on Test Data ---")
model.eval()
with torch.no_grad():
  outputs = model(X_test)
  _, predicted = torch.max(outputs.data, 1)
  y_test_np = y_test.numpy()
  predicted_np = predicted.numpy()
  accuracy = accuracy_score(y_test_np, predicted_np)
  precision = precision_score(y_test_np, predicted_np)
  recall = recall_score(y_test_np, predicted_np)
  f1 = f1_score(y_test_np, predicted_np)
  cm = confusion_matrix(y_test_np, predicted_np)
  
  print(f'Accuracy: {accuracy:.4f}')
  print(f'Precision: {precision:.4f}')
  print(f'Recall: {recall:.4f}')
  print(f'F1 Score: {f1:.4f}')
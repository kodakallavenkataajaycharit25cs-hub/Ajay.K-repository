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
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler  
import os

""" ----- Imports Done ----- """
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


class SpamDataset(Dataset):
  def __init__(self,messages,labels, word2vec_model,vector_size):
    self.messages = messages
    self.labels = labels
    self.word2vec_model = word2vec_model
    self.vector_size = vector_size

  def __len__(self):
    return len(self.messages)

  def __getitem__(self, index):
    message = self.messages.iloc[index]
    label = self.labels.iloc[index]
    tokens = tokenize_text(message)
    embeddings = get_avg_word2vec(tokens, self.word2vec_model,self.vector_size)
    embedded_tensor = torch.from_numpy(embeddings).float()
    label_tensor = torch.tensor(label, dtype= torch.long)
    return embedded_tensor, label_tensor 

class SpamTrainer:
  def __init__(self, model, train_loader, test_loader, criterion, optimizer, epochs=100): 
    self.model = model
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.criterion = criterion
    self.optimizer = optimizer
    self.epochs = epochs
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model.to(self.device)
    print(f"Using device: {self.device}")
    
  # Initialize GradScaler for mixed precision training if CUDA is available
    self.scaler = GradScaler("cuda") if self.device.type == 'cuda' else None    
  def train(self, start_epoch = 0):
    print(f"\n--- Starting Training from Epoch {start_epoch+1} ---")
    for epoch in range(start_epoch, self.epochs):
      self.model.train()
      total_loss = 0
      
      for batch_index, (data, target) in enumerate(self.train_loader):
        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        
        
        if self.scaler: 
          with autocast("cuda"):
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            self.scaler.scale(loss).backward() # Scale loss and call backward()
            self.scaler.step(self.optimizer)   
            self.scaler.update()               
        else:
          outputs = self.model(data)
          loss = self.criterion(outputs, target)
          loss.backward()
          self.optimizer.step()
        total_loss += loss.item()        
      if (epoch + 1) % 20 == 0:
        current_loss = total_loss / len(self.train_loader)
        print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {current_loss:.4f}")
        self.save_checkpoint(epoch + 1, current_loss, filename=f"spam_checkpoint_epoch_{epoch+1}.pth.tar")
    print("\nTraining finished!") 


  def evaluate(self):
    print("\n--- Evaluating on Test Data ---") 
    self.model.eval()
    all_predictions = []
    all_true_labels = [] 
    
    with torch.no_grad():
      for data, target in self.test_loader:
        data, target = data.to(self.device), target.to(self.device) 
        
        outputs = self.model(data)
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_true_labels.extend(target.cpu().numpy())

    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision = precision_score(all_true_labels, all_predictions)
    recall = recall_score(all_true_labels, all_predictions)
    f1 = f1_score(all_true_labels, all_predictions)
    cm = confusion_matrix(all_true_labels, all_predictions)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
  
  def save_checkpoint(self, epoch, loss, filename = "Checkpoint.pth.tar"):
    state = {"epoch" : epoch,
    "loss" : loss,
    "model_state_dict" : self.model.state_dict(),
    "optimizer_state_dict" : self.optimizer.state_dict()     
    }
    
    torch.save(state, filename)
    print(f"Checkpoint Saved in {filename} at Epoch : {epoch}")
  
  def load_checkpoint(self, filename = "Checkpoint.pth.tar" ):
    print(f"Loading Checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location = self.device)
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"Checkpoint loaded. Resuming from epoch {epoch}, Loss: {loss:.4f}")
    return epoch, loss
  
    
  
    

script_dir = pathlib.Path(__file__).parent.resolve()
csv_path = script_dir / 'SMSSpamCollection.csv'
df = pd.read_csv(csv_path,sep= "\t",header = None,names=["label","message"],encoding='latin-1')
df['label'] = df["label"].map({"ham" : 0 , "spam": 1})
df['message'] = df["message"].apply(preprocess_text)

X = df["message"] 
y = df["label"]
X_train_raw , X_test_raw , y_train , y_test = train_test_split(X,y,test_size= 0.2,random_state=42)


X_train_tokens_for_w2v = X_train_raw.apply(tokenize_text).tolist()

w2v_model = Word2Vec.Word2Vec(vector_size=100, window=5, min_count=1, workers=4)
w2v_model.build_vocab(X_train_tokens_for_w2v) 
w2v_model.train(X_train_tokens_for_w2v, total_examples=w2v_model.corpus_count, epochs=10)


train_dataset = SpamDataset(X_train_raw, y_train, w2v_model, vector_size=100)
test_dataset = SpamDataset(X_test_raw, y_test, w2v_model, vector_size=100)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) 

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


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

trainer = SpamTrainer(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=100 
)

checkpoint_path = "Checkpoint.pth.tar" 
start_epoch = 0
if os.path.exists(checkpoint_path):
  start_epoch, _ = trainer.load_checkpoint(checkpoint_path)

trainer.train(start_epoch = start_epoch)
trainer.evaluate()

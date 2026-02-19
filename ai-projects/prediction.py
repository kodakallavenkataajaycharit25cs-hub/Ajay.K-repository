import torch
import torch.nn as nn
from gensim.models import Word2Vec
import numpy as np
from Spam import tokenize_text, get_avg_word2vec

# --- imports ----

def predict_msg(msg, model, w2v_model):
  # Preprocess and Tokenize
  tokens = tokenize_text(msg)
  vector = get_avg_word2vec(tokens, w2v_model, 100)
  
  # Safety Guard for unknown words
  if np.all(vector == 0):
    return "Ham", 1.0

  vector_tensor = torch.from_numpy(vector).float().unsqueeze(0)
  
  model.eval()
  with torch.no_grad():
    output = model(vector_tensor)
    probs = torch.softmax(output, dim=1)
    
    # Extract probabilities
    ham_prob = probs[0][0].item()
    spam_prob = probs[0][1].item()

  # --- SPAM-HUNTER LOGIC ---
  # Expanded flags for Urgency, Phishing, and Security scams
  red_flags = [
      "claim", "offer", "winner", "urgent", "urgency", "click", "limited", "prize", "cash",
      "verify", "account", "locked", "alert", "immediately", "security", "suspended", "action required"
  ]
  msg_lower = msg.lower()
  
  # Count how many red flags appear
  found_flags = [flag for flag in red_flags if flag in msg_lower]
  
  if found_flags:
    # Base boost of 0.20, plus 0.05 for every additional flag found (Stacking Boost!)
    boost = 0.20 + (0.05 * (len(found_flags) - 1))
    spam_prob += boost
    spam_prob = min(spam_prob, 0.99)

  # Use a 40% threshold for Spam (More aggressive)
  if spam_prob > 0.40:
    label = "Spam"
    confidence = spam_prob
  else:
    label = "Ham"
    confidence = ham_prob

  return label, confidence
model = nn.Sequential(
  nn.Linear(100, 64), nn.ReLU(), nn.Dropout(0.2),
  nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
  nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2),
  nn.Linear(16, 2)
)

checkpoint = torch.load("spam_checkpoint_epoch_100.pth.tar", map_location= "cpu")
model.load_state_dict (checkpoint["model_state_dict"])
w2v_model = Word2Vec.load("word2vec.model")

#Test run
msg = "Hey man, are we still meeting at 5?"
label, conf = predict_msg(msg, model, w2v_model)
print(f"Result: {label} ({conf*100:.2f}%)")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import gradio as gr
print("Imports done!")  # Test

dataset = load_dataset("beans")  
classes = dataset['train'].features['labels'].names
print(f"Classes: {classes}")  # ['healthy', 'angular_leaf_spot', 'bacterial_blight', 'rust']
print(f"Train: {len(dataset['train'])}, Test: {len(dataset['test'])} imgs")

transform_train = transforms.Compose([
  transforms.Resize([224,224]),
  transforms.ToTensor(),
  transforms.Normalize([0.485,0.546,0.406],[0.229,0.224,0.225])
])

class BeansDataset(torch.utils.data.Dataset):
  def __init__(self,split = "train",transform = None):
    self.ds = dataset[split]
    self.transform = transform
  def __len__(self):
    return len(self.ds)
  def __getitem__(self, idx):
    img = self.ds['image'][idx]
    label = self.ds['labels'][idx]
    img = transform_train(img)  
    return img, label

train_loader = DataLoader(BeansDataset('train',transform_train),batch_size = 16,shuffle= True)
test_loader = DataLoader(BeansDataset('test',transform_train),batch_size = 16,shuffle= True)
print("Data Ready")

class CropNet(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
      self.bn1 = nn.BatchNorm2d(32)
      self.pool = nn.MaxPool2d(2)
      self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
      self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
      self.bn2 = nn.BatchNorm2d(64)      
      self.bn3 = nn.BatchNorm2d(128)     
      self.fc = nn.Linear(128 * 28 * 28, len(classes))
      
    def forward(self, x):  # x=[16,3,224,224]
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  
        x = x.view(x.size(0), -1)                        
        return self.fc(x) 
            
model = CropNet()
print(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Auto GPU!
print(f"🚀 Using: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

model.to(device)
dummy = torch.randn(1,3,224,224).to(device)
print("Forward OK:", model(dummy).shape)  # torch.Size([1, 3])
print(f"Classes: {len(classes)}, Model on {device}")
optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

imgs, labs = next(iter(train_loader))
imgs = imgs.to(device)
labs = labs.to(device)
print(f"Batch OK: {imgs.shape}")
out = model(imgs)
print(f"Sample logit: {out[0].argmax().item()}")  

for epoch in range(10):
    model.train()
    if epoch == 0:
      print("Sample probs:", F.softmax(out[0], dim=0))  # Vary(not uniform) 

    tot_loss, tot_acc = 0, 0
    for batch_idx, (imgs, labs) in enumerate(train_loader):
        imgs, labs = imgs.to(device), labs.to(device)  
        optimizer.zero_grad()  # Reset grads
        out = model(imgs)      # Forward pass
        loss = criterion(out, labs)
        loss.backward()       
        optimizer.step()      
        tot_loss += loss.item()
        tot_acc += (out.argmax(1)==labs).float().mean()
    print(f"Ep {epoch+1}: Loss {tot_loss/len(train_loader):.3f} Acc {tot_acc/len(train_loader):.2%}")
# ===== TEST ACC =====
model.eval()  
test_correct = 0
total_test = 0
with torch.no_grad():  
    for imgs, labs in test_loader:  
        imgs = imgs.to(device)
        labs = labs.to(device)
        out = model(imgs)
        pred = out.argmax(dim=1)  # Highest logit class
        test_correct += (pred == labs).sum().item()
        total_test += labs.size(0)

print(f"🎉 Test Acc: {test_correct/total_test*100:.1f}%")
torch.save(model.state_dict(), 'crop_model.pth')

remedies = {0:"Healthy!",1:"Leaf Spot: Fungicide",2:"Blight: Copper spray",3:"Rust: Neem oil"}

def diagnose(img):
    model.eval()
    img_t = transform_test(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(img_t)[0]
    pred = probs.argmax().item()
    conf = probs[pred].item()
    sev = conf * 100  
    return f"{classes[pred]}\nConf: {conf:.1%} Sev: {sev:.0f}%\n{remedies[pred]}"

torch.save(model, 'crop_doctor_full.pth')

app = gr.Interface(fn=diagnose, inputs=gr.Image(type='pil'), outputs='text', title="Crop Disease Detector")
app.launch(server_port=7860)
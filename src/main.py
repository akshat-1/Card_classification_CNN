import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score

from set_up import PlayingCardDataset
from model import CardClassifierCNN
from early_stopping import EarlyStopping
from eval import *
from params import *


#loadng dataset and tokenizing the data
dataset = PlayingCardDataset(data_dir='./dataset/train/')
data_dir = './dataset/train/'
target_to_class = {v: k for k, v in ImageFolder(data_dir).class_to_idx.items()}

#resize
transform = transforms.Compose([
    transforms.Resize((resize_h, resize_w)),
    transforms.ToTensor(),
])
dataset = PlayingCardDataset(data_dir, transform)

#setting up dataloader
dataLoader = DataLoader(dataset, batch_size=n_batches, shuffle=True)


#model
model = CardClassifierCNN(num_classes=n_classes)
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#config
train_folder = './dataset/train/'
valid_folder = './dataset/valid/'
test_folder = './dataset/test/'

train_dataset = PlayingCardDataset(train_folder, transform=transform)
val_dataset = PlayingCardDataset(valid_folder, transform=transform)
test_dataset = PlayingCardDataset(test_folder, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=n_batches, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=n_batches, shuffle=False)
test_loader = DataLoader(val_dataset, batch_size=n_batches, shuffle=False)


num_epochs = epochs
train_losses, val_losses = [], []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Optimize performance on Apple Silicon Processors
if torch.backends.mps.is_available():
    device = torch.device("mps")

print("Selected device",device)

model = CardClassifierCNN(num_classes=n_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
es = EarlyStopping(patience=3)

epoch = 0
done = False
while epoch < num_epochs and not done:
    epoch += 1
    # Training phase
    model.train()
    running_loss = 0.0
    for images, labels in train_loader: 
        # Move inputs and labels to the device
        images, labels = images.to(device), labels.to(device)     
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validation phase
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            # Move inputs and labels to the device
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
    val_loss = running_loss / len(val_loader.dataset)
    val_losses.append(val_loss)
    print(f"Epoch {epoch}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")

    # Check early stopping criteria
    done = es(model, val_loss)
    print(f"Early Stopping: {es.status}")




# Calculate the Accuracy of the Model using test data
model.eval()
true_labels = []
predicted_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)  # Get the class index with the highest probability
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predictions.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")


torch.save(model.state_dict(), './params/model.pt')

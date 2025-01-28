import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

import pandas as pd

pd.set_option('display.max_columns', None)

# Import our files

files = [f for f in os.listdir('/home/okamitah/Projects/football-prediction-project/football-prediction-project/data-analysis/ratings') if f.endswith('.csv')]
data_frames = [pd.read_csv(os.path.join('/home/okamitah/Projects/football-prediction-project/football-prediction-project/data-analysis/ratings', f)) for f in files]
full_data = pd.concat(data_frames, ignore_index=True)

# We'll add the rank difference, rating difference and result features (home - away) to make it one column insteadof two and normalize them

full_data['Rank Difference'] = full_data['Home ranking'] - full_data['Away ranking']
print(f"rankd_mean: {full_data['Rank Difference'].mean()} | rankd_std: {full_data['Rank Difference'].std()}")
full_data['Rank Difference'] = (full_data['Rank Difference'] - full_data['Rank Difference'].mean()) / full_data['Rank Difference'].std()

full_data['Rating Difference'] = full_data['Home Last Avg Rating'] - full_data['Away Last Avg Rating']
print(f"ratingd_mean: {full_data['Rating Difference'].mean()} | ratingd_std: {full_data['Rating Difference'].std()}")
full_data['Rating Difference'] = (full_data['Rating Difference'] - full_data['Rating Difference'].mean()) / full_data['Rating Difference'].std()

full_data['Results'] = full_data['Home score'] - full_data['Away score']
full_data['Label'] = full_data['Results'].apply(lambda x: 1 if x > 0 else (0 if x == 0 else 2))
full_data['Results'] = (full_data['Results'] - full_data['Results'].mean()) / full_data['Results'].std()

# And make our tensors

features = ['Rank Difference', 'Rating Difference', 'Home odds', 'Draw odds', 'Away odds']



X = full_data[features].values
y = full_data['Label'].values

print(f"rankd_mean: {full_data['Rank Difference'].mean()} | rankd_std: {full_data['Rank Difference'].std()}\nratingd_mean: {full_data['Rating Difference'].mean()} | ratingd_std: {full_data['Rating Difference'].std()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Move tensors to GPU if available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

# We'll make our Dataset class (to ensure our structure conformity)

class FootballDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = FootballDataset(X_train_tensor, y_train_tensor)
test_dataset = FootballDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# And make our model

class BettingNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(BettingNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

model = BettingNN(input_size=5, num_classes=3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0000001)

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

def evaluate_model(model, test_loader):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

num_epochs = 100000000
train_model(model, train_loader, criterion, optimizer, num_epochs)
evaluate_model(model, test_loader)

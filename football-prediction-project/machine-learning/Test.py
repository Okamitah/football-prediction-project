import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

import pandas as pd

pd.set_option('display.max_columns', None)

files = [f for f in os.listdir('/home/okamitah/git/football-prediction-project/data-analysis/ratingst') if f.endswith('.csv')]
data_frames = [pd.read_csv(os.path.join('/home/okamitah/git/football-prediction-project/data-analysis/ratingst', f)) for f in files]
full_data = pd.concat(data_frames, ignore_index=True)

full_data['Rank Difference'] = full_data['Home ranking'] - full_data['Away ranking']
full_data['Rank Difference'] = (full_data['Rank Difference'] - full_data['Rank Difference'].mean()) / full_data['Rank Difference'].std()

full_data['Rating Difference'] = full_data['Home Last Avg Rating'] - full_data['Away Last Avg Rating']
full_data['Rating Difference'] = (full_data['Rating Difference'] - full_data['Rating Difference'].mean()) / full_data['Rating Difference'].std()

full_data['Results'] = full_data['Home score'] - full_data['Away score']
full_data['Label'] = full_data['Results'].apply(lambda x: 1 if x > 0 else (0 if x == 0 else 2))
full_data['Results'] = (full_data['Results'] - full_data['Results'].mean()) / full_data['Results'].std()
odds_features = ['Home odds', 'Draw odds','Away odds']
odds = full_data[odds_features].values

features = ['Rank Difference', 'Rating Difference', 'Home odds', 'Draw odds', 'Away odds']

X = full_data[features].values
y = full_data['Label'].values

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
odds_tensor = torch.tensor(odds, dtype=torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


X_tensor = X_tensor.to(device)
y_tensor = y_tensor.to(device)
odds_tensor = odds_tensor.to(device)

class FootballDataset(Dataset):
    def __init__(self, X, y, odds):
        self.X = X
        self.y = y
        self.odds = odds

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.odds[idx]


dataset = FootballDataset(X_tensor, y_tensor, odds_tensor)

loader = DataLoader(dataset, batch_size=128, shuffle=True)

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


def custom_loss_function(predictions, labels, odds):
    probabilities = torch.softmax(predictions, dim=1)

    batch_size = labels.size(0)
    ev = probabilities * odds

    actual_ev = ev.gather(1, labels.unsqueeze(1)).squeeze()
    loss = -actual_ev.mean()

    return loss

def train_model(model, train_loader, optimizer, num_epochs):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels, odds in train_loader:

            inputs, labels, odds = inputs.to(device), labels.to(device), odds.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = custom_loss_function(outputs, labels, odds)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
          print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

def evaluate_profitability(model, test_loader):
    model.eval()
    total_bets = 0
    total_profit = 0.0

    with torch.no_grad():
        for inputs, labels, odds in test_loader:
            inputs, odds = inputs.to(device), odds.to(device)
            outputs = model(inputs)

            probabilities = torch.softmax(outputs, dim=1)
            predicted_outcomes = probabilities.argmax(dim=1)
            for i in range(inputs.size(0)):
                pred = predicted_outcomes[i].item()
                if pred == labels[i].item():
                    profit = odds[i, pred] - 1
                else:
                    profit = -1
                total_profit += profit
                total_bets += 1
    roi = (total_profit / total_bets) if total_bets > 0 else 0
    print(f"Total Bets: {total_bets}, Total Profit: {total_profit:.2f}, ROI: {roi * 100:.2f}%")

checkpoint = torch.load('checkpoint.pth', weights_only=True)
optimizer = torch.optim.Adam(model.parameters())
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']

model.eval()
evaluate_profitability(model, loader)

def find_value_bets(model, data_loader):
    model.eval()
    value_bets = []

    with torch.no_grad():
        for inputs, labels, odds in data_loader:
            inputs, odds = inputs.to(device), odds.to(device)
            outputs = model(inputs)

            # Get model probabilities
            probabilities = torch.softmax(outputs, dim=1)

            # Calculate implied probabilities
            implied_probabilities = 1 / odds

            # Check for value bets
            for i in range(inputs.size(0)):
                value_bet_found = False
                value_bet_info = {}

                for outcome in range(3):  # 3 outcomes: Home Win, Draw, Away Win
                    model_prob = probabilities[i, outcome].item()
                    implied_prob = implied_probabilities[i, outcome].item()

                    if model_prob > implied_prob:  # Value bet condition
                        value_bet_found = True
                        value_bet_info[outcome] = {
                            'model_prob': model_prob,
                            'implied_prob': implied_prob,
                            'odds': odds[i, outcome].item(),
                        }

                if value_bet_found:
                    value_bets.append({
                        'game_index': i,
                        'value_bet_info': value_bet_info,
                    })

    return value_bets

value_bets = find_value_bets(model, loader)

# Display value bets

for bet in value_bets:
    print(f"Game Index: {bet['game_index']}")
    for outcome, info in bet['value_bet_info'].items():
        outcome_label = ["Home Win", "Draw", "Away Win"][outcome]
        print(f"  Outcome: {outcome_label}")
        print(f"    Model Probability: {info['model_prob']:.2f}")
        print(f"    Implied Probability: {info['implied_prob']:.2f}")
        print(f"    Odds: {info['odds']:.2f}")

# output:
# Total Bets: 1770, Total Profit: 1345.17, ROI: 76.00%
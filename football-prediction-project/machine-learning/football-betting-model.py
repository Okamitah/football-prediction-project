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

files = [f for f in os.listdir('/home/okamitah/git/football-prediction-project/data-analysis/ratings') if f.endswith('.csv')]
data_frames = [pd.read_csv(os.path.join('/home/okamitah/git/football-prediction-project/data-analysis/ratings', f)) for f in files]
full_data = pd.concat(data_frames, ignore_index=True)

# We'll add the rank difference, rating difference and result features (home - away) to make it one column insteadof two and normalize them

full_data['Rank Difference'] = full_data['Home ranking'] - full_data['Away ranking']
full_data['Rank Difference'] = (full_data['Rank Difference'] - full_data['Rank Difference'].mean()) / full_data['Rank Difference'].std()

full_data['Rating Difference'] = full_data['Home Last Avg Rating'] - full_data['Away Last Avg Rating']
full_data['Rating Difference'] = (full_data['Rating Difference'] - full_data['Rating Difference'].mean()) / full_data['Rating Difference'].std()

full_data['Results'] = full_data['Home score'] - full_data['Away score']
full_data['Label'] = full_data['Results'].apply(lambda x: 1 if x > 0 else (0 if x == 0 else 2))
full_data['Results'] = (full_data['Results'] - full_data['Results'].mean()) / full_data['Results'].std()

# And make our tensors

features = ['Rank Difference', 'Rating Difference', 'Results', 'Home odds', 'Draw odds', 'Away odds']

X = full_data[features].values
y = full_data['Label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


print(X_test_tensor)
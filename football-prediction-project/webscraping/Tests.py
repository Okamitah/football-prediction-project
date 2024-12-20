import os
import pandas as pd

directory = 'data2'
csv_files = [file for file in os.listdir(directory)]

dataframes = {}  # Use a dictionary to store DataFrames with filenames as keys

for file in csv_files:
    file_path = os.path.join(directory, file)
    dataframes[file] = pd.read_csv(file_path)

for file in dataframes:
    #print(f"{file}: {dataframes[file].isnull().sum()}")
    print(f"{file}: {dataframes[file].isnull().sum().sum()}")
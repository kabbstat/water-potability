import numpy as np
import pandas as pd
import os

def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        return data
    except Exception as e:
        raise Exception(f"error loading data from {filepath}: {e}")
#data_train = pd.read_csv("./data/raw/train.csv")
#data_test = pd.read_csv("./data/raw/test.csv")

def fill_missing_values(data):
    try:
        for column in data.columns:
           if data[column].isnull().any():
               median_value = data[column].median()
               data[column].fillna(median_value, inplace=True)
        return data
    except Exception as e:
        raise Exception(f"error filling missing values: {e}")

def save_data(data, filepath):
    try:
        data.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"error saving data to {filepath}: {e}")

def main(): 
    data_path = os.path.join('data', 'processed')
    os.makedirs(data_path, exist_ok=True)
    data_train = load_data("./data/raw/train.csv")
    data_test = load_data("./data/raw/test.csv")
    train_processed_data = fill_missing_values(data_train)
    test_processed_data = fill_missing_values(data_test)
    save_data(train_processed_data, os.path.join(data_path, 'train_processed.csv'))
    save_data(test_processed_data, os.path.join(data_path, 'test_processed.csv'))
if __name__ == "__main__":
    main()  
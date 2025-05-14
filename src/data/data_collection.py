import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml

data = pd.read_csv(r"C:\Users\HP\OneDrive\Bureau\water_potability.csv")
def load_params(filepath):
    try:
        with open(filepath,'r') as file:
            params = yaml.safe_load(file)
        return params['data_collection']['test_size']
    except Exception as e:
        raise Exception(f"error loading parameters from {filepath}: {e}")
    
def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        return data
    except Exception as e:
      raise Exception(f"error loading data from {filepath}: {e}")


#test_size = yaml.safe_load(open('params.yaml'))['data_collection']['test_size']
def split_data(data, test_size:float):
    try:
       train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
       return train_data, test_data
    except Exception as e:
        raise Exception(f"error splitting data: {e}")
#train_data , test_data = train_test_split(data, test_size=test_size, random_state=42)
def save_data(data:pd.DataFrame, filepath:str) -> None:
    try:
       data.to_csv(filepath, index=False)
    except Exception as e:
        raise Exception(f"error saving data to {filepath}: {e}")

def main():
    test_size = load_params('params.yaml')
    data = load_data(r"C:\Users\HP\OneDrive\Bureau\water_potability.csv")
    train_data, test_data = split_data(data, test_size)
    raw_data_path = os.path.join('data', 'raw')
    os.makedirs(raw_data_path)
    save_data(train_data, os.path.join(raw_data_path, 'train.csv'))
    save_data(test_data, os.path.join(raw_data_path, 'test.csv'))

if __name__ == "__main__":
    main()


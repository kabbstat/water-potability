import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import json
import pickle
from dvclive import Live
import yaml

def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        X = data.drop(columns=['Potability'])
        y= data['Potability']
        return X, y
    except Exception as e:
        raise Exception(f'error in loading data test from {filepath}: {e}')
#data_test = pd.read_csv('./data/processed/test_processed.csv')
#X_test = data_test.drop(columns=['Potability'])
#y_test = data_test['Potability']
def predict_model(X):
    try:
        with open('models/model.pkl', 'rb') as file:
            model = pickle.load(file)
            y_pred = model.predict(X)
            return y_pred
    except Exception as e:
        raise Exception(f'error in prediction: {e}')
#model = pickle.load(open('model.pkl', 'rb'))
#y_pred = model.predict(X_test)
def evaluation(y_test, y_pred):
    try :
        params = yaml.safe_load(open('params.yaml', 'r'))
        test_size = params["data_collection"]["test_size"]
        n_estimators = params["model_building"]["n_estimators"]
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        re = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', acc)
            live.log_metric('precision', pre)
            live.log_metric('recall', re)
            live.log_metric('f1_score', f1)
            live.log_param('test_size', test_size)
            live.log_param('n_estimators', n_estimators)
        metrics ={
            'accuracy': acc,
            'precision': pre,
            'recall': re,
            'f1_score': f1
        }
        return metrics
    except Exception as e:
        raise Exception (f'error in metrics calculation: {e}')

def save_metrics(metrics, filepath):
    try:
        with open(filepath, 'w') as file:
            json.dump(metrics, file, indent=4)
    except Exception as e:
        raise Exception(f'error in saving metrics to {filepath}: {e}')


#with open('metrics.json','w') as file :
    #json.dump(metrics, file, indent = 4)
def main():
    try:
        X_test, y_test = load_data('./data/processed/test_processed.csv')
        y_pred = predict_model(X_test)
        metrics = evaluation(y_test, y_pred)
        save_metrics(metrics, 'reports/metrics.json')
    except Exception as e:
        print(f'An error occured:{e}')
if __name__=='__main__':
    main()    
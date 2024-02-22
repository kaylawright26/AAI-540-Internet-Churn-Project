import argparse
import requests
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import os
import io
import time
from time import strftime, gmtime
if __name__ == "__main__":
    base_dir = "/opt/ml/processing"

    df = pd.read_csv(
        f"{base_dir}/input/internet_churn.csv")
    
    # Fix spelling error in column
    df = df.rename(columns = {'reamining_contract':'remaining_contract'})
    df['remaining_contract'] = df['remaining_contract'].astype(str)
    
    # Fix negative values
    df = df[df[df.columns].min(axis=1) >= 0]
    
    # Discretize column
    df['remaining_contract'].replace('nan', 'no contract', inplace=True)
    for i in df['remaining_contract']:
        try:
            if float(i) >= 0 and float(i) <1:
                df['remaining_contract'].replace(i, '0-1 years', inplace=True)
            elif float(i) >= 1 and float(i) < 2:
                df['remaining_contract'].replace(i, '1-2 years', inplace=True)
            elif float(i) >= 2 and float(i)<3:
                df['remaining_contract'].replace(i, '2-3 years', inplace=True)
        except:
            continue
            
    # Fill na with column median 
    df[['download_avg','upload_avg']] = df[['download_avg','upload_avg']].fillna(df[['download_avg','upload_avg']].median())
    
    # Get dummy variables
    df = pd.get_dummies(df, columns = ['remaining_contract'],dtype = int)
    
    # Rename columns
    df= df.rename({'remaining_contract_0-1 years':'remaining_contract_0-1_years',
                  'remaining_contract_1-2 years': 'remaining_contract_1-2_years',
                  'remaining_contract_2-3 years': 'remaining_contract_2-3_years',
                  'remaining_contract_no contract':'remaining_contract_no_contract'},axis = 1)
    train, validation, test = np.split(df, [int(0.7 * len(df)), int(0.85 * len(df))])

    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
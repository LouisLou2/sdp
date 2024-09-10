import os

import arff
import pandas as pd

def printDatasetColumns():
    folder_path = '../dataset/'
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.parquet'):
            if file_name == 'PC2.parquet':
                continue
            data=pd.read_parquet(folder_path+file_name)
            print(f'{file_name} {data.columns.tolist()}')

def checkColSame(filenames):
    folder_path = '../dataset/'
    data = pd.read_parquet(folder_path+filenames[0])
    cols = data.columns.tolist()
    rows_count = data.shape[0]
    for i in range(1,len(filenames)):
        data=pd.read_parquet(folder_path+filenames[i])
        if cols != data.columns.tolist():
            print(f'Columns of {filenames[0]} and {filenames[i]} are not the same')
            return
        rows_count += data.shape[0]
    print(f'Columns of {filenames} are the same')
    print(f'Total rows count: {rows_count}')

checkColSame(['PC1.parquet','PC3.parquet','PC4.parquet','MW1.parquet'])
# printDatasetColumns()
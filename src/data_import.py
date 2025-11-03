import pandas as pd

def data_import(fileName):
    df = pd.read_csv("data/"+fileName, index_col=0)
    print(df)

    return df
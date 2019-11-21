
import os
import pandas as pd
from os.path import abspath
from os.path import join as JP
from sklearn.model_selection import train_test_split


def load_csv(path:str, name:str):
    err = 'Path does not exists: {}'.format(abspath(path))
    assert os.path.exists(path), err
    if '.csv' not in name:
        name += '.csv'
    return pd.read_csv(
        JP(path,name), sep=',', index_col=0).fillna(0)


def save_csv(df:pd.DataFrame, path:str, name:str, keep_index:bool=False): 
    # Keep Index if index is important like a DatetimeIndex
    err = 'Path does not exists: {}'.format(abspath(path))
    assert os.path.exists(path), err
    if '.csv' not in name:
        name += '.csv'
    df.to_csv(JP(path,name), index=keep_index, sep=',')
    return


def preproces(df, dt_col='CALDENDAR_MONTH'):
    err = str('{} is not a column of the dataframe').format(dt_col)
    assert dt_col in list(df), err

    df = df.sort_values(by=dt_col)
    df[dt_col] = pd.DatetimeIndex(
            pd.to_datetime(df[dt_col]).dt.date)
    df.set_index(dt_col, drop=True, inplace=True)
    return df


def dataset_split(data:pd.DataFrame, method:str='random', train_size:int=0.8):
    err = 'Methods should be either random or temporal'
    assert method in ['random', 'temporal'], err

    tr_size = int(train_size*len(data))
    X,y = data.iloc[:,:-1].values, data.iloc[:,-1].values
    if method == 'random':
        return train_test_split(X,y,train_size=0.8)
    elif method == 'temporal':
        return X[:tr_size], X[tr_size:], y[:tr_size], y[tr_size:]


def filter_dict_by_keys(d,ks):
    ''' Returns a dictionary expect the given keys '''
    if not isinstance(ks, list):
        ks = [ks]
    return {i:d[i] for i in d if i not in ks}


def filter_dict_by_vals(d,upp_thres=None,low_thres=None):
    ''' Return the dictionary expect the keys where the values
    don't reach the boudned conditions '''
    if not upp_thres and low_thres:
        return {i:d[i] for i in d if d[i] > low_thres}
    if upp_thres and not low_thres:
        return {i:d[i] for i in d if d[i] < upp_thres}
    if upp_thres and low_thres:
        return {i:d[i] for i in d if (d[i] < upp_thres and d[i] < upp_thres)}
    return d


def filter_df_mean_thres(df,low_thres=None, upp_thres=None):
    ''' Return the dictionary expect the mean of the columns
    don't reach the boudned conditions '''
    if not low_thres and upp_thres:
        return df.loc[:,df.mean(axis=0) < upp_thres]     
    if low_thres and not upp_thres:
        return df.loc[:,df.mean(axis=0) > low_thres] 
    # return df.loc[:, df.mean(axis=0).between(low_thres, upp_thres)]
    return df.loc[:,(df.mean(axis=0) > low_thres) & (df.mean(axis=0) < upp_thres)] 
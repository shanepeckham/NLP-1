
import pandas as pd


def order_df_count(df:pd.DataFrame, col=None):
    ''' Order Columns by Frequency of the Word in the Entire Corpus '''
    if not col:
        return df[df.sum().sort_values(ascending=False).index.to_list()]
    return df[df[col].sort_values(ascending=False).index.to_list()]


def check_differences(s1:list,s2:list):
    ''' Explains the diference between to list of objects '''
    sortset = lambda l: sorted(sorted(list(l)),key=len, reverse=True)
    s12 = s1 - s2
    s21 = s2 - s1
    s = s1 ^ s2
    print('Elements present in A but not in B: ', sortset(s12))
    print('Elements present in B but not in A: ', sortset(s21))
    print('Elements present in only on of them: ', sortset(s))
    return s
import pandas as pd


def summary(df,feature):
    uniq_len = len(df[feature].unique())
    print("unique number count of {} : {}\n".format(feature,uniq_len))
    if uniq_len < 20:
        print(df[feature].value_counts(dropna=False).sort_index())
    else:
        print(df[feature])
    



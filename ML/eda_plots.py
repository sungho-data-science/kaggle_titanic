import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def bar_plot_for_categorical_feature(df, feature,nrows=1,ncols=1,index=1,sharex=None,sharey=None):
    feature_df = pd.pivot_table(df, values='count', index=feature, columns='Survived',aggfunc='count', observed=False, fill_value=0).sort_index()

    feature_0 = feature_df[0]
    feature_1 = feature_df[1]

    X = np.arange(len(feature_df))

    ax = plt.subplot(nrows,ncols,index,sharex=sharex,sharey=sharey)
    ax.bar(X-0.1, feature_0, width=0.2, align='center')
    ax.bar(X+0.1, feature_1, width=0.2, align='center')
    ax.legend(('dead','alive'))
    ax.set_xticks(X, feature_df.index, rotation = 45)
    ax.set_title("plot for {}".format(feature) , fontsize=12)
    #plt.show()

    #print text
    for xpos, ypos, yval in zip(X-0.1,feature_0/2, feature_0):
        ax.text(xpos,ypos,round(yval,2),ha="center",va="center", size=8)
    for xpos, ypos, yval in zip(X+0.1,feature_1/2, feature_1):
        ax.text(xpos,ypos,round(yval,2),ha="center",va="center", size=8)

    return ax



def normalized_stacked_barplot_for_categorical_feature(df, feature,nrows=1,ncols=1,index=1,sharex=None,sharey=None):
    feature_df = pd.pivot_table(df, values='count', index=feature, columns='Survived',aggfunc='count', observed=False, fill_value=0).sort_index()

    feature_0 = np.array(feature_df[0])
    feature_1 = np.array(feature_df[1])
    sum_level = feature_0+feature_1

    #normalization
    y_0 = np.divide(feature_0*100, sum_level, out=np.zeros(feature_0.shape, dtype=float), where=sum_level!=0)
    y_1 = np.divide(feature_1*100, sum_level, out=np.zeros(feature_1.shape, dtype=float), where=sum_level!=0)

    X = np.arange(len(feature_df))

    ax = plt.subplot(nrows,ncols,index,sharex=sharex,sharey=sharey)
    ax.bar(X, y_0, width=0.2, align='center')
    ax.bar(X, y_1, bottom=y_0, width=0.2, align='center')
    ax.legend(('dead','alive'))
    ax.set_xticks(X, feature_df.index, rotation = 45)
    ax.set_title("plot for {}".format(feature) , fontsize=12)

    #print text
    for xpos, ypos, yval in zip(X,y_0/2, y_0):
        ax.text(xpos,ypos,round(yval,2),ha="center",va="center", size=8)
    for xpos, ypos, yval in zip(X,y_0+y_1/2, y_1):
        ax.text(xpos,ypos,round(yval,2),ha="center",va="center", size=8)
    for xpos, ypos, yval in zip(X,y_0+y_1, sum_level):
        ax.text(xpos,ypos,round(yval,2),ha="center",va="bottom", size=8)

    ax.set_ylim(0,110)

    return ax



def agg_plot(df, feature, figsize = (12,8)):
    plt.figure(figsize=figsize)
    ax1 = bar_plot_for_categorical_feature(df,feature,2,1,1)
    ax2 = normalized_stacked_barplot_for_categorical_feature(df,feature,2,1,2,sharex=ax1)


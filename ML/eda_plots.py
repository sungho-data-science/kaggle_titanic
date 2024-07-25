import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def bar_plot_for_categorical_feature(df, feature):
    feature_df = pd.pivot_table(df, values='count', index=feature, columns='Survived',aggfunc='count').sort_index()
    feature_0 = feature_df[0]
    feature_1 = feature_df[1]
    X = np.arange(len(feature_df))
    ax = plt.subplot(111)
    ax.bar(X-0.1, feature_0, width=0.2, align='center')
    ax.bar(X+0.1, feature_1, width=0.2, align='center')
    ax.legend(('dead','alive'))
    plt.xticks(X, feature_df.index)
    plt.title("plot for {}".format(feature) , fontsize=17)
    plt.show()
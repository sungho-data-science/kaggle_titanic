import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from datetime import datetime
import pytz

## extract current timestamp in string type
def current_timestamp():
    EST_ZONE = pytz.timezone("America/New_York") 
    CURRENT_TIME_EST = datetime.now(EST_ZONE)
    current_timestamp = CURRENT_TIME_EST.strftime("%Y%m%d-%H%M%S")
    return current_timestamp



def pipeline_preprocess(continuous, categorical):
    transformer_num = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    transformer_cate = Pipeline(
    #    steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))]
        steps=[("encoder", TargetEncoder())]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", transformer_num, continuous),
            ("cat", transformer_cate, categorical),
        ]
    )

    return preprocessor


def bayesian_opt_xgboost(X_train, y_train, pbounds, preprocessor):
    def xgboost_bayes_opt(learning_rate, n_estimators, max_depth, gamma, preprocessor = preprocessor):
        max_depth = int(max_depth)
        n_estimators = int(n_estimators)
        clf = Pipeline(
            steps=[("preprocessor", preprocessor), 
                ("classifier", XGBClassifier(
                    learning_rate = learning_rate,
                    n_estimators = n_estimators,
                    max_depth = max_depth,
                    gamma = gamma,
                    eval_metric='aucpr',
                    random_state = 42,
                    enable_categorical=True,               
                ))]
        )
        return np.mean(cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy'))

    optimizer = BayesianOptimization(
        f = xgboost_bayes_opt,
        pbounds = pbounds,
        random_state = 42
    )

    optimizer.maximize(init_points=20, n_iter=4)

    print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))

    best_param = optimizer.max["params"]
    best_param["max_depth"] = int(best_param["max_depth"])
    best_param["n_estimators"] = int(best_param["n_estimators"])

    return best_param




def xgb_fit_and_predict(preprocessor, best_param, X_train, y_train, X_test):
    model_XGB = Pipeline(
        steps=[("preprocessor", preprocessor), 
            ("classifier", XGBClassifier(
                **best_param,
                eval_metric='aucpr',
                random_state = 42,
                enable_categorical=True,               
            ))]
    )
    model_XGB.fit(X_train,y_train)

    return model_XGB.predict(X_test)


def bayesian_opt_logisticRegression(X_train, y_train, pbounds, preprocessor):
    def logistic_bayes_opt(C):
        clf = Pipeline(
            steps=[("preprocessor", preprocessor), 
                ("classifier", LogisticRegression(C = C, random_state = 42)
                )]
        )
        return np.mean(cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy'))

    optimizer = BayesianOptimization(
        f = logistic_bayes_opt,
        pbounds = pbounds,
        random_state = 42
    )

    optimizer.maximize(init_points=20, n_iter=4)

    print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))

    best_param = optimizer.max["params"]
    return best_param


def logisticRegression_fit_and_predict(preprocessor, best_param, X_train, y_train, X_test):
    model_logistic = Pipeline(
        steps=[("preprocessor", preprocessor), 
            ("classifier", LogisticRegression(**best_param, random_state = 42)
            )]
    )
    model_logistic.fit(X_train,y_train)

    return model_logistic.predict(X_test).astype(int)

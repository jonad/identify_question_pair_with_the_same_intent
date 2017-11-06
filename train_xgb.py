from collections import defaultdict

import pandas as pd

from models.xgboost_model import *
from utils.utils import split_dataset_xgb

FILE_NAME = 'data/quora_xgb.pickle'


seed = 10

def main():
    # get the data
    data = pd.read_pickle(FILE_NAME)
    dataset = data.values
    X = dataset[:, 1:]
    Y = dataset[:, 0]
    X_train, X_test, y_train, y_test = split_dataset_xgb(X, Y, 0.2, seed)
    
    # set tuning parameters
    params_grid = defaultdict(list)
    #params_grid['n_estimators'] = range(100, 1050, 50)
    params_grid['learning_rate'] = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    #params_grid['max_depth'] = range(2, 12, 2)
    # params_grid['subsample'] = [0.5, 0.75, 1.0]
    # params_grid['colsample_bytree'] = [0.4, 0.6, 0.8, 1.0]
    params_grid = dict(params_grid)

    xgb_model = XgbModel.from_params(X_train, y_train, param_grid=params_grid, num_folds=10, scoring='f1', seed=seed)
    
    # get best params
   # _, best_params = search_xgb_params(X_train, y_train, param_grid=params_grid, num_folds=10, scoring='f1', seed=seed)
    
    # train the model
    # xg_model = xgb.XGBClassifier(nthread=-1, **best_params)
    # xg_model = train(xg_model, X_train, y_train)
    xgb_model.train(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    print(y_pred[:5])
    predictions = [round(value) for value in y_pred]
    acc, f1 = xgb_model.evaluate(y_test, predictions)
    print(acc, f1)
    pickle.dump(xgb_model, open("models/quora_model.dat", "wb"))
    
    
if __name__ == '__main__':
    main()
    


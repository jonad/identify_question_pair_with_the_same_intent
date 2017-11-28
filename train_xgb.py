from collections import defaultdict

import pandas as pd

from models.xgboost_model import *


TRAIN_PATH = 'data/quora_xgb_train.pickle'
TEST_PATH = 'data/quora_xgb_test.pickle'
seed = 10

def main():
    
    # Prepare training data
    data_train = pd.read_pickle(TRAIN_PATH).values
    X_train = data_train[:, 1:]
    y_train = data_train[:, 0]
    
    # Prepare test data
    data_test = pd.read_pickle(TEST_PATH).values
    X_test = data_test[:, 1:]
    y_test = data_test[:, 0]
    
    
    # set tuning parameters
    params_grid = defaultdict(list)
    params_grid['n_estimators'] = [950, 1000, 1050]
    params_grid['learning_rate'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    params_grid['max_depth'] = [8, 10, 12]
    params_grid = dict(params_grid)
    
    # Define the model with the best parameters.
    xgb_model = XgbModel.from_params(X_train, y_train, param_grid=params_grid, num_folds=10, scoring='f1', seed=seed)
    
    ## Train the model
    xgb_model.train(X_train, y_train)

    
    # Evaluate the model
    y_pred = xgb_model.predict(X_test)
    print(y_pred[:5])
    predictions = [round(value) for value in y_pred]
    acc, f1 = xgb_model.evaluate(y_test, predictions)
    print(acc, f1)
    
    
    
if __name__ == '__main__':
    main()
    


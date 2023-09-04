from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


def train_and_evaluate(X_train, y_train, X_test, y_test):
    rf = RandomForestRegressor(n_estimators=50)

    # Fit the model to the training data
    rf.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = rf.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return mse, rmse, rf

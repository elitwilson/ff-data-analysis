from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


def load_and_split_data():
    diabetes = load_diabetes()
    X = diabetes.data  # Conventionally, X is capitalized
    y = diabetes.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    return X_train, X_test, y_train, y_test

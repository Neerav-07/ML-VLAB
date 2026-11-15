
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset():
    """
    Load the built-in Diabetes dataset.
    """

    diabetes = load_diabetes()

    X = pd.DataFrame(
        diabetes.data,
        columns=diabetes.feature_names
    )

    y = pd.Series(
        diabetes.target,
        name="Disease Progression"
    )

    return X, y


def preprocess_data(test_size=0.2, random_state=42):
    """
    Split and standardize the dataset.
    """

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    # Standardize input features
    X_scaler = StandardScaler()

    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    # Standardize target values
    y_scaler = StandardScaler()

    y_train_scaled = y_scaler.fit_transform(
        y_train.to_numpy().reshape(-1, 1)
    ).ravel()

    y_test_scaled = y_scaler.transform(
        y_test.to_numpy().reshape(-1, 1)
    ).ravel()

    return (
        X_train_scaled,
        X_test_scaled,
        y_train_scaled,
        y_test_scaled,
        y_scaler,
        X,
        y
    )

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """
    Load input data from flat file with target variable as first column followed by however
    many feature variables

    Args:
        filepath: location of flat file
        

    Returns:
        abels: list of class labels for binary classification
        features: list of string values containing names of model features
        target: list of singular string value containing target variable name
        X_train: numpy ndarray of model features training data values
        X_test: numpy ndarray of model features test data values
        y_train: numpy ndarray of model target variable training data values
        y_test: numpy ndarray of model target variable test data values
        
    """

    df = pd.read_csv(filepath)
    target = df.columns[0]
    features = [col for col in df.columns if col not in target]
    X = df[features].values
    y = df[target].values
    labels = df[target].unique().tolist()
    labels.sort()

    train_df, test_df = train_test_split(df, test_size = .30, random_state=42)


    X_train = train_df[features].values
    y_train = train_df[target].values
    X_test = test_df[features].values
    y_test = test_df[target].values


    return labels, features, target, X_train, X_test, y_train, y_test

from enum import Enum
from typing import Any

import pandas
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset: DataFrame = pandas.read_csv('breast-cancer.csv')
dataset.drop('id', axis=1, inplace=True)

data_splits = []


class DataSplit(Enum):
    TRAIN = 0
    TEST = 2
    VALIDATION = 4


class FeatureSet(Enum):
    INPUT = 0
    TARGET = 1


def get_data_split(split: DataSplit, feature: FeatureSet) -> Any:
    return data_splits[split.value + feature.value]


def visualise_dataset() -> None:
    """

    :return: None
    """
    # TODO
    pass


def preprocess_data() -> None:
    """
    Analyses the input dataset and makes it usable for a neural network system. Then proceeds to split the data into specific sets for training.

    :return: None
    """

    # Global values to be modified
    global dataset
    global data_splits

    print('Stage 1: Preprocessing data\n')
    print('--- Null data check ---')
    print(dataset.isnull().any())
    print('\n')

    print('--- Preliminary Statistics ---')
    print(dataset.describe())
    print('\n')

    print('--- Empty Column Value Count ---')
    columns_missing_data = []
    for column in dataset.columns:
        missing = dataset.loc[dataset[column] == 0].shape[0]
        if missing > 0:
            columns_missing_data.append(column)

        print(column + ": " + str(missing))

    print('\n')

    if columns_missing_data.__len__() > 0:
        print('--- Empty Column Value Replacement (Mean) ---')
        for column in columns_missing_data:
            dataset[column] = dataset[column].fillna(dataset[column].mean())
            print(f'Replacing empty values in {column} with mean of {dataset[column].mean()}')

        print('\n')

    print('--- Feature Set and Data Split ---')
    X = dataset.drop('diagnosis', axis=1)
    y = dataset['diagnosis']

    # 80% Training, 20% Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 20% Validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    print('\n')

    # Total Split: 60% Training, 20% Testing, 20% Validation

    print('--- Data Normalisation (Data Scaling) ---')

    # Data is normalised and scaled AFTER splitting to avoid data leaking
    scaler: StandardScaler = StandardScaler()
    scaler.fit(X_train)
    print('Scaled training data')
    scaler.fit(X_test)
    print('Scaled test data')
    scaler.fit(X_val)
    print('Scaled validation data')
    print('\n')

    print('--- Stage 1: Preprocessing Data - Complete ---')
    print('Caching data splits')
    data_splits = [
        X_train,
        y_train,
        X_test,
        y_test,
        X_val,
        y_val,
    ]


def create_model() -> None:
    """

    :return: None
    """
    # TODO
    pass


def run_evaluation() -> None:
    """

    :return: None
    """
    # TODO
    pass


if __name__ == "__main__":
    # TODO:
    # - Model training
    # - Testing and analysis
    #     - Notable ROC Curve

    # visualise_dataset()
    preprocess_data()
    create_model()
    run_evaluation()

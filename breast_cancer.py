from enum import Enum

import pandas
from keras import Sequential
from keras.src.layers import Dense
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset: DataFrame = pandas.read_csv('breast-cancer.csv')
dataset.drop('id', axis=1, inplace=True)

data_splits = []

model: Sequential

class DataSplit(Enum):
    TRAIN = 0
    TEST = 2
    VALIDATION = 4


class FeatureSet(Enum):
    INPUT = 0
    TARGET = 1


def _get_data_split(split: DataSplit, feature: FeatureSet) -> DataFrame:
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

    print('--- Target Feature Conversion (str -> int) ---')

    y_train.replace('M', 1, inplace=True)
    y_test.replace('M', 1, inplace=True)
    y_val.replace('M', 1, inplace=True)

    y_train.replace('B', 0, inplace=True)
    y_test.replace('B', 0, inplace=True)
    y_val.replace('B', 0, inplace=True)

    print('Replaced "M" with 1 and "B" with 0 for binary classification\n')

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
    print('Caching data splits\n')
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
    Builds the tensorflow model with the correctly configured input and output layers and compiles it.
    :return: None
    """
    global model

    print('Stage 2: Model Creation\n')

    model = Sequential()

    model.add(Dense(128, activation='relu', input_dim=30))
    model.add(Dense(64, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    print('--- Model Compilation ---')
    print('Compiling model with binary_crossentropy for binary classification')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('\n')

    print('--- Model Training ---')
    model.fit(
        _get_data_split(DataSplit.TRAIN, FeatureSet.INPUT).values,
        _get_data_split(DataSplit.TRAIN, FeatureSet.TARGET).values,
        validation_data=(
            _get_data_split(DataSplit.VALIDATION, FeatureSet.INPUT).values,
            _get_data_split(DataSplit.VALIDATION, FeatureSet.TARGET).values
        ),
        epochs=200
    )

    print('Model compilation complete, Summary:')
    model.summary()
    print('\n')

    print('--- Stage 2: Model Creation - Complete ---')
    print('\n')


def run_evaluation() -> None:
    """

    :return: None
    """
    # TODO
    pass


if __name__ == "__main__":
    # TODO:
    # - Testing and analysis
    #     - Notable ROC Curve

    # visualise_dataset()
    preprocess_data()
    create_model()
    run_evaluation()

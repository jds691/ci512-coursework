import keras
import matplotlib.pyplot as plt
import pandas
import seaborn as sns
from keras import Sequential
from keras.src.layers import Dense
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import common
from common import NeuralNetworkOptions, DataSplit, FeatureSet


class BreastCancerNeuralNetwork(common.NeuralNetwork):
    _dataset: DataFrame
    _data_splits = []

    _model: Sequential
    _model_history: keras.callbacks.History

    def __init__(self, options: NeuralNetworkOptions):
        super().__init__('breast_cancer', options)
        self._dataset = pandas.read_csv('breast-cancer.csv')
        self._dataset.drop('id', axis=1, inplace=True)

    def _get_data_split(self, split: DataSplit, feature: FeatureSet) -> DataFrame:
        return self._data_splits[split.value + feature.value]

    def visualise_dataset(self) -> None:
        print('Visualisation: Dataset plotting\n')

        self._figure_category = 'Preliminary Statistics'

        visualisation_columns = self._dataset.drop('diagnosis', axis=1).columns

        for column in visualisation_columns:
            self.add_visualisation_to_queue(plt.figure(column))
            plot = sns.histplot(data=self._dataset, x=column, hue='diagnosis', kde=True)
            plot.set_title(column)

        self.show_visualisations()
        self.close_all_visualisations()

        print('--- Visualisation: Dataset plotting - Complete ---')
        print('\n')
        self.wait_for_verification()

    def preprocess_data(self) -> None:
        print('Stage 1: Preprocessing data\n')
        print('--- Null data check ---')
        print(self._dataset.isnull().any())
        print('\n')
        self.wait_for_verification()

        print('--- Preliminary Statistics ---')
        print(self._dataset.describe())
        print('\n')
        self.wait_for_verification()

        print('--- Empty Column Value Count ---')
        columns_missing_data = []
        for column in self._dataset.columns:
            missing = self._dataset.loc[self._dataset[column] == 0].shape[0]
            if missing > 0:
                columns_missing_data.append(column)

            print(column + ": " + str(missing))

        print('\n')
        self.wait_for_verification()

        if columns_missing_data.__len__() > 0:
            print('--- Empty Column Value Replacement (Mean) ---')
            for column in columns_missing_data:
                self._dataset[column] = self._dataset[column].fillna(self._dataset[column].mean())
                print(f'Replacing empty values in {column} with mean of {self._dataset[column].mean()}')

            print('\n')
            self.wait_for_verification()

        print('--- Feature Set and Data Split ---')
        X = self._dataset.drop('diagnosis', axis=1)
        y = self._dataset['diagnosis']

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

        print('--- Target Feature Conversion (Series -> NDArray) ---')

        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        y_val = y_val.to_numpy()

        print('\n')

        print('--- Target Feature Conversion (1D -> 2D) ---')

        y_train = y_train.reshape((y_train.shape[0], 1))
        y_test = y_test.reshape((y_test.shape[0], 1))
        y_val = y_val.reshape((y_val.shape[0], 1))

        print('Reshaped target features to 2D: Shape(*, 1)\n')
        self.wait_for_verification()

        print('--- Data Normalisation (Data Scaling) ---')

        # Data is normalised and scaled AFTER splitting to avoid data leaking
        scaler: StandardScaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        print('Scaled training data')
        X_test = scaler.fit_transform(X_test)
        print('Scaled test data')
        X_val = scaler.fit_transform(X_val)
        print('Scaled validation data')
        print('\n')
        self.wait_for_verification()

        print('--- Stage 1: Preprocessing Data - Complete ---')
        print('Caching data splits\n')
        self._data_splits = [
            X_train,
            y_train,
            X_test,
            y_test,
            X_val,
            y_val,
        ]
        self.wait_for_verification()

    def create_model(self) -> None:
        print('Stage 2: Model Creation\n')

        self._model = Sequential()

        self._model.add(Dense(128, activation='relu', input_dim=30))
        self._model.add(Dense(64, activation='relu'))

        self._model.add(Dense(1, activation='sigmoid'))

        print('--- Model Compilation ---')
        print('Compiling model with binary_crossentropy for binary classification')
        self._model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[
            'accuracy',
            keras.metrics.TruePositives(),
            keras.metrics.TrueNegatives(),
            keras.metrics.FalsePositives(),
            keras.metrics.FalseNegatives(),
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            keras.metrics.F1Score(),
        ])
        print('\n')
        self.wait_for_verification()

        print('--- Model Training ---')
        self._model_history = self._model.fit(
            self._get_data_split(DataSplit.TRAIN, FeatureSet.INPUT),
            self._get_data_split(DataSplit.TRAIN, FeatureSet.TARGET),
            validation_data=(
                self._get_data_split(DataSplit.VALIDATION, FeatureSet.INPUT),
                self._get_data_split(DataSplit.VALIDATION, FeatureSet.TARGET)
            ),
            epochs=self._options.epochs,
        )
        self.wait_for_verification()

        print('Model compilation complete, Summary:')
        self._model.summary()
        print('\n')
        self.wait_for_verification()

        print('--- Stage 2: Model Creation - Complete ---')
        print('\n')
        self.wait_for_verification()

    def run_evaluation(self) -> None:
        print('Stage 3: Model Evaluation\n')

        print('--- Metrics over Epochs ---')

        self._figure_category = 'Model Metrics'

        history_frame = pandas.DataFrame(self._model_history.history)
        history_frame['epoch'] = self._model_history.epoch

        ignored_metrics = [
            'epoch',
            'val_accuracy',
            'val_f1_score',
            'val_false_negatives',
            'val_false_positives',
            'val_loss',
            'val_precision',
            'val_recall',
            'val_true_positives',
            'val_true_negatives',
        ]

        working_history_copy = history_frame.copy(deep=True)

        for metric in ignored_metrics:
            if metric in working_history_copy.columns:
                working_history_copy = working_history_copy.drop(metric, axis=1)

        for metric in working_history_copy.columns:
            self.add_visualisation_to_queue(plt.figure(metric))
            plot = sns.lineplot(data=history_frame, x='epoch', y=metric)
            plot.set_title(metric)

        self.show_visualisations()
        self.close_all_visualisations()
        self.wait_for_verification()

        print('--- Model Evaluate ---')

        self._model.evaluate(
            self._get_data_split(DataSplit.TEST, FeatureSet.INPUT),
            self._get_data_split(DataSplit.TEST, FeatureSet.TARGET)
        )

        # TODO: Implement ROC Curve

        print('--- Stage 3: Model Evaluation - Complete ---')
        print('\n')
        self.wait_for_verification()


if __name__ == "__main__":
    print('Running breast_cancer from main. Running all stages!\n')

    network: BreastCancerNeuralNetwork = BreastCancerNeuralNetwork(
        options=NeuralNetworkOptions(
            wait_for_verification=True
        )
    )

    network.run()

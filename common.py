from dataclasses import dataclass, field
from enum import Enum
from os import system, name

class DataSplit(Enum):
    """
    Represents data splits within in the dataset for model training purposes
    """
    TRAIN = 0
    TEST = 2
    VALIDATION = 4


class FeatureSet(Enum):
    """
    Represents the split between the input and output values
    """
    INPUT = 0
    TARGET = 1


class NeuralNetworkStages(Enum):
    """
    Represents the neural network stages that are common in all networks inheriting from common.NeuralNetwork
    """
    VISUALISATION = 0
    PREPROCESSING = 1
    COMPILATION = 2
    EVALUATION = 3


def _create_default_neural_network_stages() -> list[NeuralNetworkStages]:
    return [
        NeuralNetworkStages.VISUALISATION,
        NeuralNetworkStages.PREPROCESSING,
        NeuralNetworkStages.COMPILATION,
        NeuralNetworkStages.EVALUATION
    ]


@dataclass
class NeuralNetworkOptions:
    """
    Represents the options for neural networks when run
    """

    wait_for_verification: bool = False
    stages: list[NeuralNetworkStages] = field(default_factory=_create_default_neural_network_stages)


class NeuralNetwork:
    """
    Represents a neural network designed for classification based problems
    """

    _options: NeuralNetworkOptions

    def __init__(self, options: NeuralNetworkOptions):
        self._options = options

    def visualise_dataset(self) -> None:
        """
        Plots a graph of all values within the dataset, unscaled

        :return: None
        """
        print('Visualisation Stage - Not Configured')

    def preprocess_data(self) -> None:
        """
        Analyses the input dataset and makes it usable for a neural network system. Then proceeds to split the data into specific sets for training.

        :return: None
        """
        print('Stage 1: Preprocessing data - Not Configured')

    def create_model(self) -> None:
        """
        Builds the tensorflow model with the correctly configured input and output layers and compiles it.

        :return: None
        """
        print('Stage 2: Model Creation - Not Configured')

    def run_evaluation(self) -> None:
        """
        Calls evaluate on the test data and shows all the required evaluation metrics

        :return: None
        """
        print('Stage 3: Model Evaluation - Not Configured')

    def run(self) -> None:
        """
        Runs the stages of the Neural Network according to its input settings

        :return: None
        """
        for stage in self._options.stages:
            match stage:
                case NeuralNetworkStages.VISUALISATION:
                    self.visualise_dataset()
                case NeuralNetworkStages.PREPROCESSING:
                    self.preprocess_data()
                case NeuralNetworkStages.COMPILATION:
                    self.create_model()
                case NeuralNetworkStages.EVALUATION:
                    self.run_evaluation()

    def wait_for_verification(self) -> None:
        """
        If the model is configured to do so, it waits for user input before advancing its output

        :return: None
        """
        if not self._options.wait_for_verification:
            return

        input('Press any key to advance...')

        if name == 'nt':
            system('cls')
        else:
            system('clear')

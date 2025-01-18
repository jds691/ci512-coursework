from dataclasses import dataclass, field
from enum import Enum

class DataSplit(Enum):
    TRAIN = 0
    TEST = 2
    VALIDATION = 4


class FeatureSet(Enum):
    INPUT = 0
    TARGET = 1


class NeuralNetworkStages(Enum):
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
    stages: list[NeuralNetworkStages] = field(default_factory=_create_default_neural_network_stages)


class NeuralNetwork:
    _options: NeuralNetworkOptions

    def __init__(self, options: NeuralNetworkOptions):
        self._options = options

    def visualise_dataset(self) -> None:
        print('Visualisation Stage - Not Configured')

    def preprocess_data(self) -> None:
        print('Stage 1: Preprocessing data - Not Configured')

    def create_model(self) -> None:
        print('Stage 2: Model Creation - Not Configured')

    def run_evaluation(self) -> None:
        print('Stage 3: Model Evaluation - Not Configured')

    def run(self) -> None:
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

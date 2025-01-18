import time

import survey

from breast_cancer import BreastCancerNeuralNetwork
from common import NeuralNetworkOptions, NeuralNetworkStages
from pollution import AirPollutionNeuralNetwork

_neural_network_options = NeuralNetworkOptions()


def _display_main_menu():
    main_menu_options = (
        'Breast Cancer - Binary Classification',
        'Air Pollution - Multi-Class Classification',
        'Config',
        'Exit'
    )

    item_index = survey.routines.select('Main Menu:', options=main_menu_options)

    match item_index:
        case 0:
            network = BreastCancerNeuralNetwork(options=_neural_network_options)
            network.run()
            _display_main_menu()
        case 1:
            network = AirPollutionNeuralNetwork(options=_neural_network_options)
            network.run()
            _display_main_menu()
        case 2:
            _display_config_menu()
        case 3:
            # Allows it to exit without exception
            pass


def _display_config_menu():
    stages = [
        'Visualisation',
        'Pre-Processing',
        'Compilation',
        'Evaluation'
    ]

    verification_values = [
        'Off',
        'On'
    ]

    active = []
    for stage in _neural_network_options.stages:
        active.append(stage.value)

    form = {
        'Wait for Verification': survey.widgets.Select(options=verification_values),
        'Enabled Stages': survey.widgets.Basket(options=stages, active=active)
    }

    config_data = survey.routines.form('Config:', form=form)

    _neural_network_options.wait_for_verification = config_data['Wait for Verification']
    _neural_network_options.stages.clear()

    for stage in config_data['Enabled Stages']:
        _neural_network_options.stages.append(NeuralNetworkStages(stage))

    survey.printers.done('Config saved')
    time.sleep(1)
    _display_main_menu()

if __name__ == "__main__":
    _display_main_menu()

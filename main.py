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

    visualisation_values = [
        'Save to Disk',
        'Show Windows'
    ]

    active = []
    for stage in _neural_network_options.stages:
        active.append(stage.value)

    wait_for_verification_widget = survey.widgets.Select(options=verification_values)
    if _neural_network_options.wait_for_verification:
        # Moves the selection down by one
        wait_for_verification_widget.invoke(survey.core.Event.arrow_down, survey.core.ansi.Control(rune=''))

    visualisation_mode_widget = survey.widgets.Select(options=visualisation_values)
    if _neural_network_options.wait_for_verification:
        # Moves the selection down by one
        visualisation_mode_widget.invoke(survey.core.Event.arrow_down, survey.core.ansi.Control(rune=''))

    form = {
        'Wait for Verification': wait_for_verification_widget,
        'Visualisation Mode': visualisation_mode_widget,
        'Enabled Stages': survey.widgets.Basket(options=stages, active=active)
    }

    config_data = survey.routines.form('Config:', form=form)

    _neural_network_options.display_visualisations = config_data['Visualisation Mode']
    _neural_network_options.wait_for_verification = config_data['Wait for Verification']
    _neural_network_options.stages.clear()

    for stage in config_data['Enabled Stages']:
        _neural_network_options.stages.append(NeuralNetworkStages(stage))

    survey.printers.done('Config saved')
    time.sleep(1)
    _display_main_menu()

if __name__ == "__main__":
    _display_main_menu()

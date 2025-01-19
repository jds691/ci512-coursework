# CI512 Coursework

## Overview

This is an application designed to classify 2 separate databases using a Neural Network in Python.

## Datasets

- [Breast Caner - Rahma Sleam, Kaggle](https://www.kaggle.com/datasets/rahmasleam/breast-cancer)
- [Air Quality and Pollution Assessment, Mujtaba Mateen, Kaggle](https://www.kaggle.com/datasets/mujtabamatin/air-quality-and-pollution-assessment)

## Running

To run the program, activate the virtual environment and run main.py to access the main interface:

```shell
source .venv/bin/activate
python3 main.py
```

If you wish to run an algorithm directly, you can replace main.py with breast_cancer.py or pollution.py

> [!IMPORTANT]
> If you receive an error running main.py in the IDE please run it using Command Prompt, Powershell or your preferred
> terminal

### Config

- Wait for Verification (Default: Off): Will wait between steps until you want it to continue to verify output
- Visualisation Mode (Default: Save to Disk): Determines how to display visualisations
- Enabled Stages (Default: All): List of neural network stages to run **(Disabling compilation but enabling evaluation
  will crash)**

When running breast_cancer.py or pollution.py directly, all stages are enabled and wait for verification is on.
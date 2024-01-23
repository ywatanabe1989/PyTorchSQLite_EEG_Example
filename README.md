# PyTorchSQLite_EEGExample

This repository demonstrates how to use PyTorch with SQLite for efficient loading of EEG datasets. It includes scripts for downloading datasets from OpenNeuro, creating a SQLite database, and a data loader using PyTorch.

## Downloading Datasets
Run the following script to download EEG datasets from OpenNeuro:
```
$ ./download_openneuro_datasets.sh
```
- ds002778
[UC San Diego Resting State EEG Data from Patients with Parkinson's Disease](https://openneuro.org/datasets/ds002778/versions/1.0.5)
- ds003478
[EEG: Depression rest](https://openneuro.org/datasets/ds003478/versions/1.1.0/file-display/sub-002:eeg:sub-002_task-Rest_run-01_coordsystem.json)
- ds003775
[SRM Resting-state EEG](https://openneuro.org/datasets/ds003775/versions/1.2.1)
- ds004504
[A dataset of EEG recordings from: Alzheimer's disease, Frontotemporal dementia and Healthy subjects](https://openneuro.org/datasets/ds004504/versions/1.0.6)

## Setting Up the Python Environment
To set up the Python environment, follow these steps:

```
$ python -m venv env
$ source env/bin/activate
$ python -m pip install -U pip
$ python -m pip install -r requirements.txt
```

## Creating a SQLite Database
This step creates a SQLite database ([`./data/eeg_data.db`](./data/eeg_data.db)) which consolidates the downloaded OpenNeuro datasets. Run the following command:
```
$ python ./populate_SQLite_database.py | tee populate_SQLite_database.log
```

## Using SQLiteDataLoader
To load data using the SQLiteDataLoader script, execute:

```
$ python ./SQLiteDataLoader.py | tee SQLiteDataLoader.log
```

## Contact
For any inquiries or contributions, please contact ywatanabe1989@gmail.com.

## Setup

1. Install the required packages using:
    ```
    pip install -r requirements.txt
    ```
2. Create the input dataset

## Construct Graph Datasets

We construct the graph datasets by loading the csv and construct PyG graph data. We then inject anomalies into the dataset. For each dataset, please run:
- `elliptic plus plus` dataset: `python data_ellipticpp.py`

## Run Experiment

To run the experiments, please execute the corresponding file: 

1. `GrapBEAN`: 
    ```
    python train_full_experiment.py --name ellipticpp_anomaly --id 0
    ```
 
The argument `--name` indicates which dataset we want the model run on, with the format of `{dataset_name}_anomaly`. Additional arguments are also available depending on the models.

- Arguments for **all** models.
    ```
    --name              : dataset name
    --id                : which instance of anomaly injected graph [0-9]
    ```
- Arguments for `GraphBEAN`.
    ```
    --n-epoch           : number of epoch in the training [default: 50]
    --lr                : learning rate [default: 1e-2]
    --eta                     : structure decoder loss weight [default: 0.2]
    --score-agg               : aggregation method for node anomaly score
                                (max or mean) [default: max]      
    --scheduler-milestones    : milestones for learning scheduler [default: []]            
    ```

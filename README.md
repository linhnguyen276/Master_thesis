## Setup

1. Install the required packages using:
    ```
    pip install -r requirements.txt
    ```
2. Create the input dataset:
The Elliptic++ dataset consists of 203k Bitcoin transactions and 822k wallet addresses to enable both the detection of fraudulent transactions and the detection of illicit addresses (actors) in the Bitcoin network by leveraging graph data.\
\
DATASET CAN BE FOUND HERE: https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l 


## Construct Graph Datasets

We construct the graph datasets by loading the csv and construct PyG graph data. We then inject anomalies into the dataset. For each dataset, please run:
- `elliptic plus plus` dataset: `python data_ellipticpp.py`

## Run Experiment

To run the experiments, please execute the corresponding file: 

1. `Semi-supervised GrapBEAN model`: 
    ```
    python train_full_experiment.py
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
2. `Distillation algorithm for the GNN's output`: 
    ```
    python train_and_distill.py
    ```

## Citation
1. Youssef Elmougy, Ling Liu (2023). Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network for Financial Forensics. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’23), August 6–10, 2023, Long Beach, CA, USA. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3580305.3599803
2. R. Fathony, J. Ng and J. Chen (2023), "Interaction-Focused Anomaly Detection on Bipartite Node-and-Edge-Attributed Graphs," 2023 International Joint Conference on Neural Networks (IJCNN), Gold Coast, Australia, 2023, pp. 1-10, https://doi.org/10.1109/IJCNN54540.2023.10191331 
3. Wanyu Lin, Hao Lan, and Baochun Li (2021). "Generative Causal Explanations for Graph Neural Networks," in the Proceedings of the 38th International Conference on Machine Learning (ICML 2021), Online, July 18-24, 2021. https://arxiv.org/pdf/2104.06643.pdf
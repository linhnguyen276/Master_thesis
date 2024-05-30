# Copyright 2021 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

import torch
from torch_sparse.tensor import SparseTensor

import numpy as np
from anomaly_insert import inject_random_block_anomaly

from models.data import BipartiteData

import torch
from sklearn import preprocessing

import pandas as pd

# %%


def standardize(features: np.ndarray) -> np.ndarray:
    scaler = preprocessing.StandardScaler()
    z = scaler.fit_transform(features)
    return z


def prepare_data():

    cols = ["user_id", "item_id", "timestamp", "state_label"] + [
        f"v{i+1}" for i in range(151)
    ]

    df = pd.read_csv(f"data/ellipticspp_val.csv", skiprows=1, names=cols)

    # edge
    cols_d = {"item_id": [("n_action", "count")]}
    for i in range(151):
        cols_d[f"v{i+1}"] = [(f"v{i+1}_mean", "mean")]

    df_edge = df.groupby(["user_id", "item_id", "timestamp", "state_label"]).agg(cols_d)
    df_edge = df_edge.droplevel(axis=1, level=0).reset_index()
    df_edge.to_csv(f"data/ellipticsppedge_val.csv")

    # user
    cols_d = {"item_id": [("n_item", "nunique"), ("n_action", "count")]}
    for i in range(151):
        cols_d[f"v{i+1}"] = [(f"v{i+1}_mean", "mean")]

    df_user = df.groupby(["user_id", "timestamp", "state_label"]).agg(cols_d)

    df_user = df_user.droplevel(axis=1, level=0).reset_index()
    df_user.to_csv(f"data/ellipticsppaddress_val.csv")

    # item
    cols_d = {"user_id": [("n_user", "nunique"), ("n_action", "count")]}
    for i in range(151):
        cols_d[f"v{i+1}"] = [(f"v{i+1}_mean", "mean")]

    df_item = df.groupby(["item_id", "timestamp", "state_label"]).agg(cols_d)
    df_item = df_item.droplevel(axis=1, level=0).reset_index()
    df_item.to_csv(f"data/ellipticspptransaction_val.csv")


def create_graph():

    df_user = pd.read_csv("data/ellipticsppaddress_val.csv")
    df_item = pd.read_csv("data/ellipticspptransaction_val.csv")

    ### sample to do the run
    df_edge = pd.read_csv("data/ellipticsppedge_val.csv")

    df_user = df_user[df_user['user_id'].isin(df_edge['user_id'])]
    df_user.reset_index(drop=True, inplace=True)
    df_item = df_item[df_item['item_id'].isin(df_edge['item_id'])]
    df_item.reset_index(drop=True, inplace=True)

    df_user["uid"] = df_user.index
    df_item["iid"] = df_item.index

    df_user_id = df_user[["user_id", "uid"]]
    df_item_id = df_item[["item_id", "iid"]]

    df_edge_2 = df_edge.merge(
        df_user_id,
        on="user_id",
    ).merge(df_item_id, on="item_id")
    df_edge_2 = df_edge_2.sort_values(["uid", "iid"])
#    df_edge_2.to_csv('df_edge_2.csv')

    uid = torch.tensor(df_edge_2["uid"].to_numpy())
    iid = torch.tensor(df_edge_2["iid"].to_numpy())

    adj = SparseTensor(row=uid, col=iid)
#    edge_attr = torch.tensor(standardize(df_edge_2.iloc[:, 4:-2].to_numpy())).float()
#    print(adj)
    edge_attr = torch.tensor(standardize(df_edge_2.iloc[:, 5:-2].to_numpy())).float()
    e_label = torch.tensor(df_edge_2.iloc[:,4].to_numpy()).float()

    user_attr = torch.tensor(standardize(df_user.iloc[:, 4:-1].to_numpy())).float()
    product_attr = torch.tensor(standardize(df_item.iloc[:, 4:-1].to_numpy())).float()
    u_label = torch.tensor(df_user.iloc[:, 3].to_numpy()).float()
    v_label = torch.tensor(df_item.iloc[:, 3].to_numpy()).float()

    data = BipartiteData(adj, xu=user_attr, xv=product_attr, xe=edge_attr, u_pred=u_label, v_pred=v_label, e_pred=e_label)

    return data


def store_graph(name: str, dataset):
    torch.save(dataset, f"storage/{name}.pt")


def load_graph(name: str, key: str, id=None):
    if id == None:
        data = torch.load(f"storage/{name}.pt")
        return data[key]
    else:
        data = torch.load(f"storage/{name}.pt")
        return data[key][id]


def synth_random():
    # generate nd store data
    import argparse

    parser = argparse.ArgumentParser(description="GraphBEAN")
    parser.add_argument("--name", type=str, default="ellipticpp_anomaly_val", help="name")
    parser.add_argument("--n-graph", type=int, default=1, help="n graph")

    args = vars(parser.parse_args())

    prepare_data()
    graph = create_graph()
    store_graph("ellipticpp-graph-val", graph)
    # graph = torch.load(f'storage/ellipticpp-graph.pt')

    graph_anomaly_list = []
    for i in range(args["n_graph"]):
        print(f"GRAPH ANOMALY {i} >>>>>>>>>>>>>>")
        print(graph)
        graph_multi_dense = inject_random_block_anomaly(
            graph, num_group=20, num_nodes_range=(1, 20), num_nodes_range2=(1, 6)
        )
        graph_anomaly_list.append(graph_multi_dense)
        print()

    dataset = {"args": args, "graph_val": graph, "graph_anomaly_list_val": graph_anomaly_list}

    store_graph(args["name"], dataset)


if __name__ == "__main__":
    synth_random()

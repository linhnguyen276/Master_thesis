from torch_sparse.tensor import SparseTensor
import numpy as np
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

    cols = ["user_id", "item_id", "timestamp", "class_address", "class_tx"] + [
        f"v{i+1}" for i in range(151)
    ]

    df = pd.read_csv(f"data/Node668_k10_re/Distillation_transaction_4hops.csv", skiprows=1, names=cols)

    # edge
    cols_d = {"item_id": [("n_action", "count")]}
    for i in range(151):
        cols_d[f"v{i+1}"] = [(f"v{i+1}_mean", "mean")]

    df_edge = df.groupby(["user_id", "item_id", "timestamp", "class_address"]).agg(cols_d)
    df_edge = df_edge.droplevel(axis=1, level=0).reset_index()
    df_edge.to_csv(f"data/Node668_k10_re/Distillation_ellipticspp_edge_train.csv")

    # user
    cols_d = {"item_id": [("n_item", "nunique"), ("n_action", "count")]}
    for i in range(151):
        cols_d[f"v{i+1}"] = [(f"v{i+1}_mean", "mean")]

    df_user = df.groupby(["user_id", "timestamp", "class_address"]).agg(cols_d)

    df_user = df_user.droplevel(axis=1, level=0).reset_index()

    df_user.to_csv(f"data/Node668_k10_re/Distillation_ellipticspp_address_train.csv")

    # item
    cols_d = {"user_id": [("n_user", "nunique"), ("n_action", "count")]}
    for i in range(151):
        cols_d[f"v{i+1}"] = [(f"v{i+1}_mean", "mean")]

    df_item = df.groupby(["item_id", "timestamp", "class_tx"]).agg(cols_d)
    df_item = df_item.droplevel(axis=1, level=0).reset_index()
    df_item.to_csv(f"data/Node668_k10_re/Distillation_ellipticspp_transaction_train.csv")


def create_graph():

    df_user = pd.read_csv("data/Node668_k6_re/Distillation_ellipticspp_address_train.csv", na_values=['N/A'])
    df_item = pd.read_csv("data/Node668_k6_re/Distillation_ellipticspp_transaction_train.csv", na_values=['N/A'])

    ### sample to do the run
    df_edge = pd.read_csv("data/Node668_k6_re/Distillation_ellipticspp_edge_train.csv")

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

    #For Gem model
    # df_item["iid"] = df_item["iid"] + len(df_user["uid"])

    df_user_id = df_user[["user_id", "uid"]]
    df_item_id = df_item[["item_id", "iid"]]

    df_edge_2 = df_edge.merge(
        df_user_id,
        on="user_id",
    ).merge(df_item_id, on="item_id")
    df_edge_2 = df_edge_2.sort_values(["uid", "iid"])
    df_edge_2 = df_edge_2.drop_duplicates()

    uid = torch.tensor(df_edge_2["uid"].to_numpy())
    iid = torch.tensor(df_edge_2["iid"].to_numpy())

    adj = SparseTensor(row=uid, col=iid)

    # adj1 = SparseTensor(row=uid, col=iid)
    # adj1_dense = adj1.to_dense()
    # size = adj1.sparse_sizes()
    # adj1_extended = torch.zeros((size[1], size[1]))
    # adj1_extended[:size[0],:] = adj1_dense
    #
    # adj2 = SparseTensor(row=iid, col=uid)
    # adj2_dense = adj2.to_dense()
    # size = adj2.sparse_sizes()
    # adj2_extended = torch.zeros((size[0], size[0]))
    # adj2_extended[:, :size[1]] = adj2_dense
    #
    # adj_dense = adj1_extended + adj2_extended
    # row, col = adj_dense.nonzero(as_tuple=True)
    # adj_values = adj_dense[row, col]
    # adj_extended = SparseTensor(row=row, col=col, value=adj_values, sparse_sizes=(size[0], size[0]))

    #
    # # Back to GraphBEAN model
    edge_attr = torch.tensor(standardize(df_edge_2.iloc[:, 5:-2].to_numpy())).float()
    e_label = torch.tensor(df_edge_2.iloc[:, 4].to_numpy()).float()
    df_edge_2.iloc[:, -1] = df_edge_2.iloc[:, -1] + len(df_user["uid"])
    e_index = torch.tensor(df_edge_2.iloc[:, -2:].to_numpy()).float()

    user_attr = torch.tensor(standardize(df_user.iloc[:, 4:-1].to_numpy())).float()
    product_attr = torch.tensor(standardize(df_item.iloc[:, 4:-1].to_numpy())).float()
    u_label = torch.tensor(df_user.iloc[:, 3].to_numpy()).float()
    v_label = torch.tensor(df_item.iloc[:, 3].to_numpy()).float()

    data = BipartiteData(adj,xu=user_attr,xv=product_attr, xe=edge_attr,u_pred=u_label, v_pred=v_label, e_pred=e_label, e_index=e_index)
    # data = BipartiteData(adj_extended, xu=user_attr, xv=product_attr, xe=edge_attr, u_pred=u_label, v_pred=v_label,
    #                      e_pred=e_label, e_index=e_index)

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
    # parser.add_argument("--name", type=str, default="Gem_val", help="name")
    parser.add_argument("--n-graph", type=int, default=1, help="n graph")

    args = vars(parser.parse_args())
    #
    # prepare_data()
    graph = create_graph()
    store_graph("Distillation_train", graph)

    graph_anomaly_list = []
    for i in range(args["n_graph"]):
        print(f"GRAPH ANOMALY {i} >>>>>>>>>>>>>>")
        print(graph)
        graph_multi_dense = BipartiteData(graph.adj, xu=graph.xu, xv=graph.xv, xe=graph.xe, u_pred=graph.u_pred,
                                          v_pred=graph.v_pred, e_pred=graph.e_pred, e_index=graph.e_index)
        graph_anomaly_list.append(graph_multi_dense)
        print()

    dataset = {"args": args, "graph_train": graph, "graph_anomaly_list_train": graph_anomaly_list}

    store_graph("Distillation_train", dataset)


if __name__ == "__main__":
    synth_random()

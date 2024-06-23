# Copyright 2021 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file
import csv
import sys
import pandas as pd
import networkx as nx
from torch_sparse.tensor import SparseTensor

from utils.seed import seed_all
import utils.io_utils as io_utils

from data_finefoods import load_graph
from models.score import compute_evaluation_metrics

import time
from tqdm import tqdm
import argparse
import os
import numpy as np

#from torch.utils.tensorboard import SummaryWriter
import datetime

import torch

from models.data import BipartiteData
from models.net import GraphBEAN
from models.sampler import EdgePredictionSampler
from models.loss import reconstruction_loss
from models.score import compute_anomaly_score, edge_prediction_metric, node_prediction_metric

from utils.seed import seed_all

# %% args

parser = argparse.ArgumentParser(description="GraphBEAN")
parser.add_argument("--name", type=str, default="ellipticpp_anomaly", help="name")
parser.add_argument(
    "--key", type=str, default="graph_anomaly_list", help="key to the data"
)
parser.add_argument("--id", type=int, default=0, help="id to the data")
parser.add_argument("--n-epoch", type=int, default=200, help="number of epoch")
parser.add_argument(
    "--scheduler-milestones",
    nargs="+",
    type=int,
    default=[],
    help="scheduler milestone",
)
parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
parser.add_argument("--score-agg", type=str, default="mean", help="aggregation for node anomaly score")
parser.add_argument("--eta", type=float, default=1, help="structure loss weight")

args1 = vars(parser.parse_args())

args2 = {
    "hidden_channels": 32,
    "latent_channels_u": 153,
    "latent_channels_v": 153,
    "edge_pred_latent": 32,
    "n_layers_encoder": 2,
    "n_layers_decoder": 2,
    "n_layers_mlp": 2,
    "dropout_prob": 0.0,
    "gamma": 0.2,
    "xe_loss_weight": 1.0,
    "classification_loss_weight": 1.0,
    "structure_loss_weight": args1["eta"],
    "structure_loss_weight_anomaly_score": args1["eta"],
    "iter_check": 10,
    "seed": 0,
    "neg_sampler_mult": 5,
    "k_check": 15,
    "tensorboard": False,
    "progress_bar": True,
    "n_hops": 5,
    "top_k": 10,
    "threshold": 10,
    "output": "Ellipticspp"
}

args = {**args1, **args2}

seed_all(args["seed"])

result_dir = "results/"

# %% train data
data = load_graph("ellipticpp_anomaly_train", "graph_anomaly_list_train")

u_ch = data.xu.shape[1]
v_ch = data.xv.shape[1]
e_ch = data.xe.shape[1]

print(
    f"Training data dimension: U node = {data.xu.shape}; V node = {data.xv.shape}; E edge = {data.xe.shape}; \n"
)

# %% validation data
data_val = load_graph("ellipticpp_anomaly_val", "graph_anomaly_list_val")

u_ch_val = data_val.xu.shape[1]
v_ch_val = data_val.xv.shape[1]
e_ch_val = data_val.xe.shape[1]


print(
    f"Validation data dimension: U node = {data_val.xu.shape}; V node = {data_val.xv.shape}; E edge = {data_val.xe.shape}; \n"
)

# %% test data
data_test = load_graph("ellipticpp_anomaly_test", "graph_anomaly_list_test")
data_test_extended = load_graph("Distillation_extended_train", "graph_anomaly_list_train")


u_ch_test = data_test.xu.shape[1]
v_ch_test = data_test.xv.shape[1]
e_ch_test = data_test.xe.shape[1]


print(
    f"Test data dimension: U node = {data_test.xu.shape}; V node = {data_test.xv.shape}; E edge = {data_test.xe.shape}; \n"
)

# %% model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GraphBEAN(
    in_channels=(u_ch, v_ch, e_ch),
    # in_channels_u=(u_ch, v_ch, e_ch),
    # in_channels_v=(u_ch, v_ch, e_ch),
    hidden_channels=args["hidden_channels"],
    latent_channels=(args["latent_channels_u"],args["latent_channels_v"]),
    # hidden_channels_u=args["hidden_channels_u"],
    # hidden_channels_v=args["hidden_channels_v"],
    # latent_channels_u=args["latent_channels_u"],
    # latent_channels_v=args["latent_channels_v"],
    edge_pred_latent=args["edge_pred_latent"],
    node_pred_latent=153,
    n_layers_encoder=args["n_layers_encoder"],
    n_layers_decoder=args["n_layers_decoder"],
    # n_layers_mlp=args["n_layers_mlp"],
    n_layers_mlp_node=4,
    n_layers_mlp_edge=4,
    dropout_prob=args["dropout_prob"],
    )

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=args["scheduler_milestones"], gamma=args["gamma"]
)

xu, xv = data.xu.to(device), data.xv.to(device)
xe, adj = data.xe.to(device), data.adj.to(device)
#yu, yv, ye = data.yu.to(device), data.yv.to(device), data.ye.to(device)
yu, yv, ye = data.u_pred.to(device), data.v_pred.to(device), data.e_pred.to(device)
node_pred = torch.cat((yu, yv), dim=0)

xu_val, xv_val = data_val.xu.to(device), data_val.xv.to(device)
xe_val, adj_val = data_val.xe.to(device), data_val.adj.to(device)
#yu, yv, ye = data.yu.to(device), data.yv.to(device), data.ye.to(device)
yu_val, yv_val, ye_val = data_val.u_pred.to(device), data_val.v_pred.to(device), data_val.e_pred.to(device)


xu_test, xv_test = data_test.xu.to(device), data_test.xv.to(device)
xe_test, adj_test = data_test.xe.to(device), data_test.adj.to(device)
#yu, yv, ye = data.yu.to(device), data.yv.to(device), data.ye.to(device)
yu_test, yv_test, ye_test = data_test.u_pred.to(device), data_test.v_pred.to(device), data_test.e_pred.to(device)

# eid_test = data_test.e_index.to(device)
feat_test = torch.cat((xu_test, xv_test), dim=0)
node_label_test = torch.cat((yu_test, yv_test), dim=0)

adj_test_extended = data_test_extended.adj.to(device)

# sampler
sampler = EdgePredictionSampler(adj, mult=args["neg_sampler_mult"])
sampler_val = EdgePredictionSampler(adj_val, mult=args["neg_sampler_mult"])
sampler_test = EdgePredictionSampler(adj_test, mult=args["neg_sampler_mult"])

print(args)
print()

# %% train
def train(epoch):

    model.train()

    edge_pred_samples = sampler.sample()

    optimizer.zero_grad()
    out = model(xu, xv, xe, adj, edge_pred_samples)

    loss, loss_ext, loss_component = reconstruction_loss(
        xu,
        xv,
        xe,
        adj,
        edge_pred_samples,
       # node_pred,
        yu,
        yv,
        out,
        xe_loss_weight=args["xe_loss_weight"],
        structure_loss_weight=args["structure_loss_weight"],
    )

    loss.backward()
    optimizer.step()
    scheduler.step()

    epred_metric = edge_prediction_metric(edge_pred_samples, out["eprob"])
    npred_metric = node_prediction_metric(yu, yv, out["nprob_u"], out["nprob_v"])

    return loss, loss_component, npred_metric, epred_metric

# %% validation
def val(epoch):

    start = time.time()

    # negative sampling
    edge_pred_samples = sampler_val.sample()

    with torch.no_grad():

        out = model(xu_val, xv_val, xe_val, adj_val, edge_pred_samples)

        loss, loss_ext, loss_component = reconstruction_loss(
            xu_val,
            xv_val,
            xe_val,
            adj_val,
            edge_pred_samples,
            yu_val,
            yv_val,
            out,
            xe_loss_weight=args["xe_loss_weight"],
            classification_loss_weight=args["classification_loss_weight"],
            structure_loss_weight=args["structure_loss_weight"],
        )

        npred_metric = node_prediction_metric(yu_val, yv_val, out["nprob_u"], out["nprob_v"])
        epred_metric = edge_prediction_metric(edge_pred_samples, out["eprob"])

        anomaly_score = compute_anomaly_score(
            xu_val,
            xv_val,
            xe_val,
            adj_val,
            edge_pred_samples,
            out,
            xe_loss_weight=args["xe_loss_weight"],
            structure_loss_weight=args["structure_loss_weight_anomaly_score"],
        )

        validation_metrics = compute_evaluation_metrics(
            anomaly_score, yu_val, yv_val, ye_val, agg=args["score_agg"]
        )

    elapsed = time.time() - start

    # print(
    #     f"Validation, loss: {loss:.4f}, "
    #         + f"xe: {loss_component['xe']:.4f}, "
    #         + f"[Eval acc_u: {npred_metric['acc_u']:.3f}, f1_u: {npred_metric['f1_u']:.3f} -> "
    #         + f"prec_u: {npred_metric['prec_u']:.3f}, rec_u: {npred_metric['rec_u']:.3f}] "
    #         + f"[Eval acc_v: {npred_metric['acc_v']:.5f}, f1_v: {npred_metric['f1_v']:.5f} -> "
    #         + f"prec_v: {npred_metric['prec_v']:.5f}, rec_v: {npred_metric['rec_v']:.5f}] "
    #         + f"u auc-roc: {validation_metrics['u_roc_auc']:.4f}, v auc-roc: {validation_metrics['v_roc_auc']:.4f}, e auc-roc: {validation_metrics['e_roc_auc']:.4f}, "
    #         + f"u auc-pr {validation_metrics['u_pr_auc']:.4f}, v auc-pr {validation_metrics['v_pr_auc']:.4f}, e auc-pr {validation_metrics['e_pr_auc']:.4f} "
    #         + f"[Eval acc_e: {epred_metric['acc_e']:.5f}, f1_e: {epred_metric['f1_e']:.5f} -> "
    #         + f"prec_e: {epred_metric['prec_e']:.5f}, rec_e: {epred_metric['rec_e']:.5f}] "
    #         + f"> {elapsed:.2f}s"
    # )

    return loss, loss_component, npred_metric, epred_metric

# %% evaluate and store
def eval(epoch):

    # model.eval()

    start = time.time()

    # negative sampling
    edge_pred_samples = sampler_test.sample()

    with torch.no_grad():

        out = model(xu_test, xv_test, xe_test, adj_test, edge_pred_samples)

        loss, loss_ext, loss_component = reconstruction_loss(
            xu_test,
            xv_test,
            xe_test,
            adj_test,
            edge_pred_samples,
            yu_test,
            yv_test,
            out,
            xe_loss_weight=args["xe_loss_weight"],
            classification_loss_weight=args["classification_loss_weight"],
            structure_loss_weight=args["structure_loss_weight"],
        )

        temp_tensor = out["nprob_u"][:, 1].numpy()
        np.savetxt("nprob_u_test.csv", temp_tensor, delimiter=",")

        temp_tensor = out["nprob_v"][:, 1].numpy()
        np.savetxt("nprob_v_test.csv", temp_tensor, delimiter=",")

        epred_metric = edge_prediction_metric(edge_pred_samples, out["eprob"])
        npred_metric = node_prediction_metric(yu_test, yv_test, out["nprob_u"], out["nprob_v"])
        prob1_test = torch.cat((out["nprob_u"][:, 1], out["nprob_v"][:, 1]), dim=0)
        prob0_test = 1 - prob1_test
        prob_test = torch.stack((prob0_test, prob1_test), dim=1)

    print(
        f"Validation, loss: {loss:.4f}, "
            + f"xe: {loss_component['xe']:.4f}, "
            + f"[Eval acc_u: {npred_metric['acc_u']:.3f}, f1_u: {npred_metric['f1_u']:.3f} -> "
            + f"prec_u: {npred_metric['prec_u']:.3f}, rec_u: {npred_metric['rec_u']:.3f}] "
            + f"[Eval acc_v: {npred_metric['acc_v']:.5f}, f1_v: {npred_metric['f1_v']:.5f} -> "
            + f"prec_v: {npred_metric['prec_v']:.5f}, rec_v: {npred_metric['rec_v']:.5f}] "
            # + f"u auc-roc: {validation_metrics['u_roc_auc']:.4f}, v auc-roc: {validation_metrics['v_roc_auc']:.4f}, e auc-roc: {validation_metrics['e_roc_auc']:.4f}, "
            # + f"u auc-pr {validation_metrics['u_pr_auc']:.4f}, v auc-pr {validation_metrics['v_pr_auc']:.4f}, e auc-pr {validation_metrics['e_pr_auc']:.4f} "
            + f"[Eval acc_e: {epred_metric['acc_e']:.5f}, f1_e: {epred_metric['f1_e']:.5f} -> "
            + f"prec_e: {epred_metric['prec_e']:.5f}, rec_e: {epred_metric['rec_e']:.5f}] "
    )

    dense_tensor = adj_test_extended.to_dense()
    expanded_tensor = dense_tensor.unsqueeze(0)
    adj_test_extended_array = expanded_tensor.numpy()

    feat_tensor = feat_test.unsqueeze(0)
    feat_test_array = feat_tensor.numpy()

    label_tensor = node_label_test.unsqueeze(0)
    label_test_array = label_tensor.numpy()

    prob_tensor = prob_test.unsqueeze(0)
    prob_test_array = prob_tensor.numpy()

    cg_data = {
        "xu": xu_test,
        "xv": xv_test,
        "xe": xe_test,
        "A": adj_test,
        "edge_pred_samples": edge_pred_samples,
        "adj": adj_test_extended_array,
        "feat": feat_test_array,
        "label": label_test_array,
        "pred": prob_test_array,
        "loss_extended": loss_ext,
        "train_idx": list(range(614)),
        "val_idx": list(range(614)),
        "test_idx": list(range(614)),
    }

    print("Saving current results...")
    io_utils.save_checkpoint(model, optimizer, args, num_epochs=-1, cg_dict=cg_data)

    return loss, loss_ext, loss_component, npred_metric, epred_metric, cg_data


# %% run training
loss_hist_train = []
loss_component_hist_train = []
epred_metric_hist_train = []
npred_metric_hist_train = []

# %% run validation
loss_hist_val = []
loss_component_hist_val = []
epred_metric_hist_val = []
npred_metric_hist_val = []

# %% run evaluation
loss_hist_test = []
loss_component_hist_test = []
epred_metric_hist_test = []
npred_metric_hist_test = []

# val(0)
loss_test, loss_test_ext, loss_component_test, npred_metric_test, edge_metric_test, cg_data_test = eval(0)

def get_edges(adj_dict, edge_dict, node, hop, edges=set(), visited=set()):
    for neighbor in adj_dict[node]:
        edges.add(edge_dict[node, neighbor])
        visited.add(neighbor)
    if hop <= 1:
        return edges, visited
    for neighbor in adj_dict[node]:
        edges, visited = get_edges(adj_dict, edge_dict, neighbor, hop - 1, edges, visited)
    return edges, visited

def extract(node):
    # weights = loss_diff_t[node].detach().numpy()
    weights = loss_diff_t[node].detach().numpy()
    sub_edge_idxs, visited = get_edges(adj_dict, edge_dict, node, args["n_hops"], edges=set(), visited=set({node}))

    sub_edge_idxs = np.array(list(sub_edge_idxs))
    sub_weights = weights[sub_edge_idxs.astype(np.int64)]
    edges = []
    for e in sub_edge_idxs:
        x, y = sorted_edges[e]
        edges.append((x, y))
        G[x][y]['weight'] = weights[e]
    sub_G = G.edge_subgraph(edges)
    sorted_idxs = np.argsort(sub_weights)
    edges = [edges[sorted_idx] for sorted_idx in sorted_idxs]
    sub_edge_idxs = sub_edge_idxs[sorted_idxs]

    top_k = args["top_k"]

    node_loss = loss_test_ext[node].item()
    best_loss = node_loss
    sub_G_y = sub_G.copy()

    for idx, e in enumerate(sub_edge_idxs):
        sub_G_y.remove_edge(*sorted_edges[e])
        largest_cc = max(nx.connected_components(sub_G_y), key=len)     # returns the largest connected component of the graph A in terms of the number of nodes
        sub_G2 = sub_G_y.subgraph(largest_cc)       # returns a subgraph of sub_G_y that includes only the nodes in the largest connected component and the edges between those nodes
        sub_G3 = nx.Graph()
        sub_G3.add_nodes_from(list(G.nodes))        # sub_G3 has all nodes in G but fewer edges
        sub_G3.add_edges_from(sub_G2.edges)

        rows_to_keep = [index for index, row in enumerate(eid_test) if tuple(row.tolist()) in sub_G3.edges]
        xe_short = xe_test[rows_to_keep]
        eid_test_short = eid_test[rows_to_keep]
        eid = pd.DataFrame(eid_test_short.numpy(), columns=['uid', 'iid'])
        #
        # # Extract unique values and their indices for each column
        unique_indices_uid = {val: eid.index[eid['uid'] == val].tolist() for val in eid['uid'].unique()}
        unique_indices_iid = {val: eid.index[eid['iid'] == val].tolist() for val in eid['iid'].unique()}

        # # Convert dictionaries to pandas Series for better visualization
        series_indices_uid = pd.Series(list(unique_indices_uid.keys()))
        series_indices_iid = pd.Series(list(unique_indices_iid.keys()))

        xu_short = xu_test[series_indices_uid.tolist()]
        yu_short = yu_test[series_indices_uid.tolist()]
        temp = [x-xu_test.shape[0] for x in series_indices_iid.tolist()]
        xv_short = xv_test[temp]
        yv_short = yv_test[temp]

        masked_adj = torch.from_numpy(nx.to_numpy_array(sub_G3, weight=None)).unsqueeze(0).float()
        temp = masked_adj[0]
        A_extracted = temp[:xu_test.shape[0], xu_test.shape[0]:(xu_test.shape[0] + xv_test.shape[0])]

        # Create the SparseTensor
        # Identify the non-zero elements
        indices = A_extracted.nonzero(as_tuple=False).t()
        values = A_extracted[A_extracted != 0]

        # Convert indices and values to the appropriate types for SparseTensor
        # row = indices[0]
        # col = indices[1]
        unique_row_indices, row_indices_new = torch.unique(indices[0], return_inverse=True)
        unique_col_indices, col_indices_new = torch.unique(indices[1], return_inverse=True)
        value = values.float()

        # Define the size of the SparseTensor
        sparse_sizes = (xu_short.shape[0], xv_short.shape[0])
        # sparse_sizes = (xu_test.shape[0], xv_test.shape[0])

        # Create the SparseTensor
        A_converted_back = SparseTensor(row=row_indices_new, col=col_indices_new, value=value, sparse_sizes=sparse_sizes)
        sampler_short = EdgePredictionSampler(A_converted_back, mult=args["neg_sampler_mult"])
        edge_pred_short = sampler_short.sample()

        m_out = model(xu_short, xv_short, xe_short, A_converted_back, edge_pred_short)

        m_loss, m_loss_ext, m_loss_component = reconstruction_loss(xu_short, xv_short, xe_short, A_converted_back,
                                                                   edge_pred_short, yu_short, yv_short, m_out,
                                                                   xe_loss_weight=args["xe_loss_weight"],
                                                                   classification_loss_weight=args["classification_loss_weight"],
                                                                   structure_loss_weight=args["structure_loss_weight"])

        x, y = sorted_edges[e]
        # if m_loss_ext[node] > best_loss:
        #     sub_G_y.add_edge(*sorted_edges[e])
        #     sub_G_y[x][y]['weight'] = (m_loss_ext[node] - best_loss).item()
        #     sub_G_y[x][y]
        # else:
        #     best_loss = m_loss_ext[node]
        if m_loss > best_loss:
            sub_G_y.add_edge(*sorted_edges[e])
            sub_G_y[x][y]['weight'] = (m_loss - best_loss).item()
            sub_G_y[x][y]
        else:
            best_loss = m_loss


    d = nx.get_edge_attributes(sub_G_y, 'weight')

    if d and top_k is not None:
        edges, weights = zip(*{k: d[k] for k in sorted(d, key=d.get)}.items())
        sorted_weight_idxs = np.argsort(weights)
        for idx, sorted_idx in enumerate(sorted_weight_idxs):
            sub_G_y.remove_edge(*edges[sorted_idx])
            largest_cc = max(nx.connected_components(sub_G_y), key=len)
            sub_G2 = sub_G_y.subgraph(largest_cc)
            if sub_G2.number_of_edges() < top_k:
                sub_G_y.add_edge(*edges[sorted_idx])

    save_dict = {
        "adj": np.asarray(nx.to_numpy_array(sub_G, weight=None)),
        "adj_y": nx.to_numpy_array(sub_G_y),
        "mapping": np.asarray(list(sub_G.nodes)),
        "weights": np.asarray(list(weights)),
        "label": np.asarray([cg_data_test["label"][0][n] for n in sub_G]),
        "features": feat[0][list(sub_G.nodes)]
    }
    assert save_dict['adj'].shape[0] == save_dict['adj_y'].shape[0], "{}, {}".format(save_dict['adj'].shape[0],
                                                                                     save_dict['adj_y'].shape[
                                                                                         0])
    assert save_dict['adj'].shape[0] == save_dict['mapping'].shape[0]
    torch.save(save_dict, "distillation/%s/node_%d.ckpt" % (args["output"], node))


for epoch in range(args["n_epoch"]+1):

    start = time.time()
    # loss, loss_component, epred_metric = train(epoch)
    loss, loss_component, npred_metric_train, edge_metric_train = train(epoch)
    elapsed = time.time() - start

    # epred_metric_hist.append(epred_metric)

    # print(
    #     f"#{epoch:3d}, "
    #     # + f"Loss: {loss:.4f} => xu: {loss_component['xu']:.4f}, xv: {loss_component['xv']:.4f}, "
    #     # + f"xe: {loss_component['xe']:.4f}, "
    #     # + f"e: {loss_component['e']:.4f} -> "
    #     + f"[acc_u: {npred_metric_train['acc_u']:.3f}, f1_u: {npred_metric_train['f1_u']:.3f} -> "
    #     + f"prec_u: {npred_metric_train['prec_u']:.3f}, rec_u: {npred_metric_train['rec_u']:.3f}] "
    #     + f"[acc_v: {npred_metric_train['acc_v']:.5f}, f1_v: {npred_metric_train['f1_v']:.5f} -> "
    #     + f"prec_v: {npred_metric_train['prec_v']:.5f}, rec_v: {npred_metric_train['rec_v']:.5f}] "
    #     + f"[acc: {edge_metric_train['acc_e']:.3f}, f1: {edge_metric_train['f1_e']:.3f} -> "
    #     + f"prec: {edge_metric_train['prec_e']:.3f}, rec: {edge_metric_train['rec_e']:.3f}] "
    #     + f"> {elapsed:.2f}s"
    # )


    # npred_metric_hist_test.append([npred_metric_test['f1_u'], npred_metric_test['f1_v'], edge_metric_test['f1_e']])

    if epoch % args["iter_check"] == 0:  # and epoch != 0:
        # tb eval
        npred_metric_hist_train.append([npred_metric_train['f1_u'], npred_metric_train['f1_v'], edge_metric_train['f1_e']])

        loss_val, loss_component_val, npred_metric_val, edge_metric_val = val(epoch)
        npred_metric_hist_val.append([npred_metric_val['f1_u'], npred_metric_val['f1_v'], edge_metric_val['f1_e']])

        loss_test, loss_test_ext, loss_component_test, npred_metric_test, edge_metric_test, cg_data_test = eval(epoch)
        npred_metric_hist_test.append([npred_metric_test['f1_u'], npred_metric_test['f1_v'], edge_metric_test['f1_e']])

# model_path = r'C:\PC\Linh\MSc in Data Science\Program\Thesis\Codes\GraphBEAN_new\ckpt\Gem_elliptics.pth.tar'
# model_optimal = torch.load(model_path)
#
# model.load_state_dict(model_optimal["model_state"])
# cg_data_test = model_optimal['cg']
# loss_test_ext = cg_data_test['loss_extended']
# eval(200)
#
# feat = torch.from_numpy(cg_data_test["feat"]).float()
# adj_array = torch.from_numpy(cg_data_test["adj"]).float()
# adj_np = adj_test.to_dense().numpy()
# num_nodes_set1, num_nodes_set2 = adj_np.shape
# full_adj_matrix = np.zeros((num_nodes_set1 + num_nodes_set2, num_nodes_set1 + num_nodes_set2))
# # Node 0 refers to index 0 in address/wallet dataset
#
# # Place A and its transpose in the full adjacency matrix
# full_adj_matrix[:num_nodes_set1, num_nodes_set1:] = adj_np
# full_adj_matrix[num_nodes_set1:, :num_nodes_set1] = adj_np.T
#
# label = torch.from_numpy(cg_data_test["label"]).long()
# edge_pred = cg_data_test["edge_pred_samples"]
#
# G = nx.from_numpy_array(full_adj_matrix)
# masked_loss = []
# sorted_edges = sorted(G.edges)
#
# edge_dict = np.zeros(adj_array.shape[1:], dtype=np.int64)
# adj_dict = {}
#
# for node in G:
#     adj_dict[node] = list(G.neighbors(node))
#
# for edge_idx, (x, y) in enumerate(sorted_edges[:-4]):
#     xe_short = torch.cat((xe_test[:edge_idx], xe_test[edge_idx + 1:]))
#     edge_dict[x, y] = edge_idx
#     edge_dict[y, x] = edge_idx
#     masked_adj = torch.from_numpy(full_adj_matrix).float()
#     masked_adj[x, y] = 0
#     masked_adj[y, x] = 0
#
#     # Extract the top-right submatrix which corresponds to the original bipartite adjacency matrix
#     A_extracted = masked_adj[:xu_test.shape[0], xu_test.shape[0]:(xu_test.shape[0] + xv_test.shape[0])]
#
#     # Create the SparseTensor
#     # Identify the non-zero elements
#     indices = A_extracted.nonzero(as_tuple=False).t()
#     values = A_extracted[A_extracted != 0]
#
#     # Convert indices and values to the appropriate types for SparseTensor
#     row = indices[0]
#     col = indices[1]
#     value = values.float()
#
#     # Define the size of the SparseTensor
#     sparse_sizes = (xu_test.shape[0], xv_test.shape[0])
#
#     # Create the SparseTensor
#     A_converted_back = SparseTensor(row=row, col=col, value=value, sparse_sizes=sparse_sizes)
#     sampler_short = EdgePredictionSampler(A_converted_back, mult=args["neg_sampler_mult"])
#     edge_pred_short = sampler_short.sample()
#
#     # Access the indices of the sparse tensor
#
#     m_out = model(xu_test, xv_test, xe_short, A_converted_back, edge_pred_short)
#     m_loss, m_loss_ext, m_loss_component = reconstruction_loss(
#         xu_test,
#         xv_test,
#         xe_short,
#         A_converted_back,
#         edge_pred_short,
#         yu_test,
#         yv_test,
#         m_out,
#         xe_loss_weight=args["xe_loss_weight"],
#         classification_loss_weight=args["classification_loss_weight"],
#         structure_loss_weight=args["structure_loss_weight"],
#     )
#     masked_loss += [m_loss_ext]
#
# masked_loss = torch.stack(masked_loss)
# loss_diff = masked_loss - loss_test_ext
# loss_diff_t = loss_diff.t()
#
# graphs = []
#
# with torch.no_grad():
#     for node in G:
#         # print("node in G: ", node)
#         # extract(node)
#         try:
#             extract(node)
#         except RuntimeError:
#             print("A runtime error occurs!!!")
#             continue
#
#         except IndexError:
#             print("An index error occurs!!!")
#
#         except AssertionError:
#             print("An asseration error occurs!!!")
#             continue

# %% write the npred_metric of training set, validation, test set to .csv
rows = zip(npred_metric_hist_train, npred_metric_hist_val, npred_metric_hist_test)
filename = 'training_validation_testing.csv'
with open(filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(rows)

print(f'Data has been writen to {filename}')
import csv
from models.score import compute_evaluation_metrics
import time
import argparse
import os
import numpy as np
import datetime
import torch

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
parser.add_argument("--eta", type=float, default=0.2, help="structure loss weight")

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
}

args = {**args1, **args2}
seed_all(args["seed"])
result_dir = "results/"

def load_graph(name: str, key: str):
    data = torch.load(f"storage/{name}.pt")
    return data[key][0]

# %% train data
data = load_graph("ellipticpp_anomaly_train", "graph_anomaly_list_train", args["id"])

u_ch = data.xu.shape[1]
v_ch = data.xv.shape[1]
e_ch = data.xe.shape[1]

print(
    f"Training data dimension: U node = {data.xu.shape}; V node = {data.xv.shape}; E edge = {data.xe.shape}; \n"
)

# %% validation data
data_val = load_graph("ellipticpp_anomaly_val", "graph_anomaly_list_val", args["id"])

u_ch_val = data_val.xu.shape[1]
v_ch_val = data_val.xv.shape[1]
e_ch_val = data_val.xe.shape[1]

print(
    f"Validation data dimension: U node = {data_val.xu.shape}; V node = {data_val.xv.shape}; E edge = {data_val.xe.shape}; \n"
)

# %% test data
data_test = load_graph("ellipticpp_anomaly_test", "graph_anomaly_list_test", args["id"])

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
    hidden_channels=args["hidden_channels"],
    latent_channels=(args["latent_channels_u"],args["latent_channels_v"]),
    edge_pred_latent=args["edge_pred_latent"],
    node_pred_latent=153,
    n_layers_encoder=args["n_layers_encoder"],
    n_layers_decoder=args["n_layers_decoder"],
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
yu, yv, ye = data.u_pred.to(device), data.v_pred.to(device), data.e_pred.to(device)
node_pred = torch.cat((yu, yv), dim=0)

xu_val, xv_val = data_val.xu.to(device), data_val.xv.to(device)
xe_val, adj_val = data_val.xe.to(device), data_val.adj.to(device)
yu_val, yv_val, ye_val = data_val.u_pred.to(device), data_val.v_pred.to(device), data_val.e_pred.to(device)

xu_test, xv_test = data_test.xu.to(device), data_test.xv.to(device)
xe_test, adj_test = data_test.xe.to(device), data_test.adj.to(device)
yu_test, yv_test, ye_test = data_test.u_pred.to(device), data_test.v_pred.to(device), data_test.e_pred.to(device)

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
    loss, loss_component = reconstruction_loss(
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

        loss, loss_component = reconstruction_loss(
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

    print(
        f"Validation, loss: {loss:.4f}, "
            + f"xe: {loss_component['xe']:.4f}, "
            + f"[Eval acc_u: {npred_metric['acc_u']:.3f}, f1_u: {npred_metric['f1_u']:.3f} -> "
            + f"prec_u: {npred_metric['prec_u']:.3f}, rec_u: {npred_metric['rec_u']:.3f}] "
            + f"[Eval acc_v: {npred_metric['acc_v']:.5f}, f1_v: {npred_metric['f1_v']:.5f} -> "
            + f"prec_v: {npred_metric['prec_v']:.5f}, rec_v: {npred_metric['rec_v']:.5f}] "
            + f"u auc-roc: {validation_metrics['u_roc_auc']:.4f}, v auc-roc: {validation_metrics['v_roc_auc']:.4f}, e auc-roc: {validation_metrics['e_roc_auc']:.4f}, "
            + f"u auc-pr {validation_metrics['u_pr_auc']:.4f}, v auc-pr {validation_metrics['v_pr_auc']:.4f}, e auc-pr {validation_metrics['e_pr_auc']:.4f} "
            + f"[Eval acc_e: {epred_metric['acc_e']:.5f}, f1_e: {epred_metric['f1_e']:.5f} -> "
            + f"prec_e: {epred_metric['prec_e']:.5f}, rec_e: {epred_metric['rec_e']:.5f}] "
            + f"> {elapsed:.2f}s"
    )

    return loss, loss_component, npred_metric, epred_metric

# %% evaluate and store
def eval(epoch):

    # model.eval()

    start = time.time()

    # negative sampling
    edge_pred_samples = sampler_test.sample()

    with torch.no_grad():

        out = model(xu_test, xv_test, xe_test, adj_test, edge_pred_samples)

        loss, loss_component = reconstruction_loss(
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

        epred_metric = edge_prediction_metric(edge_pred_samples, out["eprob"])
        npred_metric = node_prediction_metric(yu_test, yv_test, out["nprob_u"], out["nprob_v"])

        anomaly_score = compute_anomaly_score(
            xu_test,
            xv_test,
            xe_test,
            adj_test,
            edge_pred_samples,
            out,
            xe_loss_weight=args["xe_loss_weight"],
            structure_loss_weight=args["structure_loss_weight_anomaly_score"],
        )

        eval_metrics = compute_evaluation_metrics(
            anomaly_score, yu_test, yv_test, ye_test, agg=args["score_agg"]
        )

    elapsed = time.time() - start

    temp_tensor = out["nprob_u"][:,1].numpy()
    np.savetxt("nprob_u_test.csv", temp_tensor, delimiter=",")

    temp_tensor = anomaly_score["u_score_edge_mean"].numpy()
    np.savetxt("u_score_edge_mean_test.csv", temp_tensor, delimiter=",")

    temp_tensor = out["eprob"].numpy()
    np.savetxt("eprob_test.csv", temp_tensor, delimiter=",")

    temp_tensor = anomaly_score["e_score"].numpy()
    np.savetxt("edge_score_test.csv", temp_tensor, delimiter=",")

    print(
        f"Eval, loss: {loss:.4f}, "
            + f"xe: {loss_component['xe']:.4f}, "
            + f"[Eval acc_u: {npred_metric['acc_u']:.3f}, f1_u: {npred_metric['f1_u']:.3f} -> "
            + f"prec_u: {npred_metric['prec_u']:.3f}, rec_u: {npred_metric['rec_u']:.3f}] "
            + f"[Eval acc_v: {npred_metric['acc_v']:.5f}, f1_v: {npred_metric['f1_v']:.5f} -> "
            + f"prec_v: {npred_metric['prec_v']:.5f}, rec_v: {npred_metric['rec_v']:.5f}] "
            + f"u auc-roc: {eval_metrics['u_roc_auc']:.4f}, v auc-roc: {eval_metrics['v_roc_auc']:.4f}, e auc-roc: {eval_metrics['e_roc_auc']:.4f}, "
            + f"u auc-pr {eval_metrics['u_pr_auc']:.4f}, v auc-pr {eval_metrics['v_pr_auc']:.4f}, e auc-pr {eval_metrics['e_pr_auc']:.4f} "
            + f"[Eval acc_e: {epred_metric['acc_e']:.5f}, f1_v: {epred_metric['f1_e']:.5f} -> "
            + f"prec_v: {epred_metric['prec_e']:.5f}, rec_v: {epred_metric['rec_e']:.5f}] "
            + f"> {elapsed:.2f}s"
    )

    model_stored = {
        "args": args,
        "loss": loss,
        "loss_component": loss_component,
        "epred_metric": epred_metric,
        "npred_metric": npred_metric,
        "eval_metrics": eval_metrics,
        # "loss_hist": loss_hist_test,
        # "loss_component_hist": loss_component_hist_test,
        # "epred_metric_hist": epred_metric_hist,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    output_stored = {"args": args, "out": out, "anomaly_score": anomaly_score}

    print("Saving current results...")
    torch.save(
        model_stored,
        os.path.join(
            result_dir,
            f"graphbean-ellipticpp-eta-{args['eta']}-structure-model.th",
        ),
    )
    torch.save(
        output_stored,
        os.path.join(
            result_dir,
            f"graphbean-ellipticpp-{args['id']}-eta-{args['eta']}-structure-output.th",
        ),
    )

    return loss, loss_component, npred_metric, epred_metric


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


# tensor board
if args["tensorboard"]:
    log_dir = (
        "/logs/tensorboard/"
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        + "-"
        + args["name"]
    )
   # tb = SummaryWriter(log_dir=log_dir, comment=args["name"])
check_counter = 0

val(0)
eval(0)

for epoch in range(args["n_epoch"]+1):

    start = time.time()
    loss, loss_component, npred_metric_train, edge_metric_train = train(epoch)
    elapsed = time.time() - start

    print(
        f"#{epoch:3d}, "
        + f"[acc_u: {npred_metric_train['acc_u']:.3f}, f1_u: {npred_metric_train['f1_u']:.3f} -> "
        + f"prec_u: {npred_metric_train['prec_u']:.3f}, rec_u: {npred_metric_train['rec_u']:.3f}] "
        + f"[acc_v: {npred_metric_train['acc_v']:.5f}, f1_v: {npred_metric_train['f1_v']:.5f} -> "
        + f"prec_v: {npred_metric_train['prec_v']:.5f}, rec_v: {npred_metric_train['rec_v']:.5f}] "
        + f"[acc: {edge_metric_train['acc_e']:.3f}, f1: {edge_metric_train['f1_e']:.3f} -> "
        + f"prec: {edge_metric_train['prec_e']:.3f}, rec: {edge_metric_train['rec_e']:.3f}] "
        + f"> {elapsed:.2f}s"
    )

    if epoch % args["iter_check"] == 0:  # and epoch != 0:
        # tb eval
        npred_metric_hist_train.append([npred_metric_train['f1_u'], npred_metric_train['f1_v'], edge_metric_train['f1_e']])

        loss_val, loss_component_val, npred_metric_val, edge_metric_val = val(epoch)
        npred_metric_hist_val.append([npred_metric_val['f1_u'], npred_metric_val['f1_v'], edge_metric_val['f1_e']])

        loss_test, loss_component_test, npred_metric_test, edge_metric_test = eval(epoch)
        npred_metric_hist_test.append([npred_metric_test['f1_u'], npred_metric_test['f1_v'], edge_metric_test['f1_e']])

# %% write the npred_metric of training set, validation, test set to .csv
rows = zip(npred_metric_hist_train, npred_metric_hist_val, npred_metric_hist_test)
filename = 'training_validation_testing_output_u_nodropout.csv'
with open(filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(rows)

print(f'Data has been writen to {filename}')
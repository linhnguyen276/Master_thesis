# Copyright 2021 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

from typing import Dict
import torch
from torch import Tensor
import numpy as np

from torch_scatter import scatter
from torch_sparse import SparseTensor

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
    auc,
)
import pandas as pd


def compute_anomaly_score(
    xu: Tensor,
    xv: Tensor,
    xe: Tensor,
    adj: SparseTensor,
    edge_pred_samples: SparseTensor,
    out: Dict[str, Tensor],
    xe_loss_weight: float = 1.0,
    structure_loss_weight: float = 1.0,
) -> Dict[str, Tensor]:

    # node error, use RMSE instead of MSE
    xu_error = torch.sqrt(torch.mean((xu - out["xu"]) ** 2, dim=1))
    xv_error = torch.sqrt(torch.mean((xv - out["xv"]) ** 2, dim=1))

    # edge error, use RMSE instead of MSE
    xe_error = torch.sqrt(torch.mean((xe - out["xe"]) ** 2, dim=1))

    # edge prediction cross entropy
    edge_ce = -torch.log(out["eprob"][edge_pred_samples.storage.value() > 0] + 1e-12)

    # edge score
    e_score = xe_loss_weight * xe_error + structure_loss_weight * edge_ce

    # edge score
    u_score_edge_max = xu_error + scatter(
        e_score, adj.storage.row(), dim=0, reduce="max"
    )
    v_score_edge_max = xv_error + scatter(
        e_score, adj.storage.col(), dim=0, reduce="max"
    )
    u_score_edge_mean = xu_error + scatter(
        e_score, adj.storage.row(), dim=0, reduce="mean"
    )
    v_score_edge_mean = xv_error + scatter(
        e_score, adj.storage.col(), dim=0, reduce="mean"
    )
    u_score_edge_sum = xu_error + scatter(
        e_score, adj.storage.row(), dim=0, reduce="sum"
    )
    v_score_edge_sum = xv_error + scatter(
        e_score, adj.storage.col(), dim=0, reduce="sum"
    )

    anomaly_score = {
        "xu_error": xu_error,
        "xv_error": xv_error,
        "xe_error": xe_error,
        "edge_ce": edge_ce,
        "e_score": e_score,
        "u_score_edge_max": u_score_edge_max,
        "u_score_edge_mean": u_score_edge_mean,
        "u_score_edge_sum": u_score_edge_sum,
        "v_score_edge_max": v_score_edge_max,
        "v_score_edge_mean": v_score_edge_mean,
        "v_score_edge_sum": v_score_edge_sum,
    }

    return anomaly_score


def edge_prediction_metric(
    edge_pred_samples: SparseTensor, edge_prob: Tensor
) -> Dict[str, float]:

    edge_pred = (edge_prob >= 0.5).int().cpu().numpy()
    edge_gt = (edge_pred_samples.storage.value() > 0).int().cpu().numpy()

    acc = accuracy_score(edge_gt, edge_pred)
    prec = precision_score(edge_gt, edge_pred)
    rec = recall_score(edge_gt, edge_pred)
    f1 = f1_score(edge_gt, edge_pred)

    result = {"acc_e": acc, "prec_e": prec, "rec_e": rec, "f1_e": f1}
    return result

def node_prediction_metric(
    #node_pred: Tensor, node_prob: Tensor
    u_label: Tensor,
    v_label: Tensor,
    u_prob: Tensor,
    v_prob: Tensor
) -> Dict[str, float]:
    predicted_label_u = torch.argmax(u_prob, dim=1)
    predicted_label_v = torch.argmax(v_prob, dim=1)

    is_valid = ~torch.isnan(u_label)  # Boolean mask where True indicates valid values
    u_label_fixed = u_label.clone().detach()  # Make a copy to modify
    u_label_fixed[torch.isnan(u_label)] = 0  # Replace N/A with 0 in labels

    # Modify corresponding indices in nprob_u
    predicted_label_u = predicted_label_u.clone().detach()  # Clone to modify safely
    predicted_label_u[~is_valid] = 0  # Set probabilities to 0 where label was N/A

    is_valid = ~torch.isnan(v_label)  # Boolean mask where True indicates valid values
    v_label_fixed = v_label.clone().detach()  # Make a copy to modify
    v_label_fixed[torch.isnan(v_label)] = 0  # Replace N/A with 0 in labels

    # Modify corresponding indices in nprob_u
    predicted_label_v = predicted_label_v.clone().detach()  # Clone to modify safely
    predicted_label_v[~is_valid] = 0  # Set probabilities to 0 where label was N/A

    acc_u = accuracy_score(u_label.numpy(), predicted_label_u.numpy())
    prec_u = precision_score(u_label.numpy(), predicted_label_u.numpy())
    rec_u = recall_score(u_label.numpy(), predicted_label_u.numpy())
    f1_u = f1_score(u_label.numpy(), predicted_label_u.numpy())

    acc_v = accuracy_score(v_label.numpy(), predicted_label_v.numpy())
    prec_v = precision_score(v_label.numpy(), predicted_label_v.numpy())
    rec_v = recall_score(v_label.numpy(), predicted_label_v.numpy())
    f1_v = f1_score(v_label.numpy(), predicted_label_v.numpy())

    result = {"acc_u": acc_u, "prec_u": prec_u, "rec_u": rec_u, "f1_u": f1_u, "acc_v": acc_v, "prec_v": prec_v, "rec_v": rec_v, "f1_v": f1_v}
    return result

def compute_evaluation_metrics(
    anomaly_score: Dict[str, Tensor], yu: Tensor, yv: Tensor, ye: Tensor, agg="mean"
):
    # node u
    u_roc_curve = roc_curve(
        yu.cpu().numpy(), anomaly_score[f"u_score_edge_{agg}"].cpu().numpy()
    )
    u_pr_curve = precision_recall_curve(
        yu.cpu().numpy(), anomaly_score[f"u_score_edge_{agg}"].cpu().numpy()
    )
    u_roc_auc = auc(u_roc_curve[0], u_roc_curve[1])
    u_pr_auc = auc(u_pr_curve[1], u_pr_curve[0])

    # node v
    v_roc_curve = roc_curve(
        yv.cpu().numpy(), anomaly_score[f"v_score_edge_{agg}"].cpu().numpy()
    )
    v_pr_curve = precision_recall_curve(
        yv.cpu().numpy(), anomaly_score[f"v_score_edge_{agg}"].cpu().numpy()
    )
    v_roc_auc = auc(v_roc_curve[0], v_roc_curve[1])
    v_pr_auc = auc(v_pr_curve[1], v_pr_curve[0])

    # nedge
    e_roc_curve = roc_curve(ye.cpu().numpy(), anomaly_score["xe_error"].cpu().numpy())
    e_pr_curve = precision_recall_curve(
        ye.cpu().numpy(), anomaly_score["xe_error"].cpu().numpy()
    )
    e_roc_auc = auc(e_roc_curve[0], e_roc_curve[1])
    e_pr_auc = auc(e_pr_curve[1], e_pr_curve[0])

    metrics = {
        "u_roc_curve": u_roc_curve,
        "u_pr_curve": u_pr_curve,
        "u_roc_auc": u_roc_auc,
        "u_pr_auc": u_pr_auc,
        "v_roc_curve": v_roc_curve,
        "v_pr_curve": v_pr_curve,
        "v_roc_auc": v_roc_auc,
        "v_pr_auc": v_pr_auc,
        "e_roc_curve": e_roc_curve,
        "e_pr_curve": e_pr_curve,
        "e_roc_auc": e_roc_auc,
        "e_pr_auc": e_pr_auc,
    }

    return metrics


def attach_anomaly_score(
    anomaly_score: Dict[str, Tensor],
    dfu_id: pd.DataFrame,
    dfv_id: pd.DataFrame,
    dfe_id: pd.DataFrame,
):

    dfu_id = dfu_id.assign(
        xu_error=anomaly_score["xu_error"].cpu().numpy(),
        u_score_edge_max=anomaly_score["u_score_edge_max"].cpu().numpy(),
        u_score_edge_mean=anomaly_score["u_score_edge_mean"].cpu().numpy(),
    )

    dfv_id = dfv_id.assign(
        xv_error=anomaly_score["xv_error"].cpu().numpy(),
        v_score_edge_max=anomaly_score["v_score_edge_max"].cpu().numpy(),
        v_score_edge_mean=anomaly_score["v_score_edge_mean"].cpu().numpy(),
    )

    dfe_id = dfe_id.assign(
        xe_error=anomaly_score["xe_error"].cpu().numpy(),
        edge_ce=anomaly_score["edge_ce"].cpu().numpy(),
        e_score=anomaly_score["e_score"].cpu().numpy(),
    )

    return dfu_id, dfv_id, dfe_id

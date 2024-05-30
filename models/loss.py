# Copyright 2021 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

from typing import Dict, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from torch_sparse import SparseTensor


def reconstruction_loss(
    xu: Tensor,
    xv: Tensor,
    xe: Tensor,
    adj: SparseTensor,
    edge_pred_samples: SparseTensor,
    #node_pred: Tensor,
    u_label: Tensor,
    v_label: Tensor,
    out: Dict[str, Tensor],
    xe_loss_weight: float = 1.0,
    classification_loss_weight: float = 1.0,
    structure_loss_weight: float = 1.0,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    # feature mse
    xu_loss = F.mse_loss(xu, out["xu"])
    xv_loss = F.mse_loss(xv, out["xv"])
    xe_loss = F.mse_loss(xe, out["xe"])
    feature_loss = xu_loss + xv_loss + xe_loss_weight * xe_loss

    # classification loss
    classification_loss_u = F.binary_cross_entropy(out["nprob_u"][:,1], Tensor(u_label.numpy().astype(float)))
    classification_loss_v = F.binary_cross_entropy(out["nprob_v"][:,1], Tensor(v_label.numpy().astype(float)))

    # # structure loss
    edge_gt = (edge_pred_samples.storage.value() > 0).float()
    structure_loss = F.binary_cross_entropy(out["eprob"], edge_gt)

    loss = feature_loss + structure_loss_weight * structure_loss + classification_loss_u + classification_loss_v
    print("total loss: ", loss, ", feature loss: ", feature_loss, ", structure loss: ", structure_loss, ", classification loss u: ", classification_loss_u)

    loss_component = {
        "xu": xu_loss,
        "xv": xv_loss,
        "xe": xe_loss,
        "n": classification_loss_u,
        #"e": structure_loss,
        "total": loss,
    }

    return loss, loss_component

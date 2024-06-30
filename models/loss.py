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
    xe_loss_weight,
    classification_loss_weight: float = 1.0,
    structure_loss_weight: float = 1.0,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    # feature mse
    xu_loss = F.mse_loss(xu, out["xu"])
    xv_loss = F.mse_loss(xv, out["xv"])
    xe_loss = F.mse_loss(xe, out["xe"])
    feature_loss = xu_loss + xv_loss + xe_loss_weight * xe_loss
    xu_loss_full = F.mse_loss(xu, out["xu"], reduction='none')
    xv_loss_full = F.mse_loss(xv, out["xv"], reduction='none')


    # classification loss
    # Preprocess to handle N/A values
    is_valid = ~torch.isnan(u_label)  # Boolean mask where True indicates valid values
    u_label_fixed = u_label.clone().detach()  # Make a copy to modify
    u_label_fixed[torch.isnan(u_label)] = 0  # Replace N/A with 0 in labels

    # Modify corresponding indices in nprob_u
    out["nprob_u"][:, 1] = out["nprob_u"][:, 1].clone().detach()  # Clone to modify safely
    out["nprob_u"][~is_valid, 1] = 0  # Set probabilities to 0 where label was N/A

    is_valid = ~torch.isnan(v_label)  # Boolean mask where True indicates valid values
    v_label_fixed = v_label.clone().detach()  # Make a copy to modify
    v_label_fixed[torch.isnan(v_label)] = 0  # Replace N/A with 0 in labels

    # Modify corresponding indices in nprob_u
    out["nprob_v"][:, 1] = out["nprob_v"][:, 1].clone().detach()  # Clone to modify safely
    out["nprob_v"][~is_valid, 1] = 0  # Set probabilities to 0 where label was N/A

    # Calculate the loss
    classification_loss_u_full = F.binary_cross_entropy(out["nprob_u"][:,1], Tensor(u_label_fixed.numpy().astype(float)), reduction='none')
    classification_loss_v_full = F.binary_cross_entropy(out["nprob_v"][:,1], Tensor(v_label_fixed.numpy().astype(float)), reduction='none')
    classification_loss_u = F.binary_cross_entropy(out["nprob_u"][:,1], Tensor(u_label_fixed.numpy().astype(float)))
    classification_loss_v = F.binary_cross_entropy(out["nprob_v"][:,1], Tensor(v_label_fixed.numpy().astype(float)))

    # # structure loss
    edge_gt = (edge_pred_samples.storage.value() > 0).float()
    structure_loss = F.binary_cross_entropy(out["eprob"], edge_gt)

    loss = feature_loss + structure_loss_weight * structure_loss #+ classification_loss_u + classification_loss_v
    loss_ext = torch.cat((classification_loss_u_full, classification_loss_v_full), dim=0) # + torch.cat((xu_loss_full, xv_loss_full), dim=0)

    loss_component = {
        "xu": xu_loss,
        "xv": xv_loss,
        "xe": xe_loss,
      #  "n": classification_loss_u,
        #"e": structure_loss,
        "total": loss,
    }

    return loss, loss_ext, loss_component

# Copyright 2021 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

import torch
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor
from torch_sparse.tensor import SparseTensor


class BipartiteData(Data):
    def __init__(
        self,
        adj: SparseTensor,
        xu: OptTensor = None,
        xv: OptTensor = None,
        xe: OptTensor = None,
        u_pred: OptTensor = None,
        v_pred: OptTensor = None,
        e_pred: OptTensor = None,
        **kwargs
    ):
        super().__init__()
        self.adj = adj
        self.xu = xu
        self.xv = xv
        self.xe = xe
        self.u_pred = u_pred
        self.v_pred = v_pred
        self.e_pred = e_pred

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __inc__(self, key, value, *args, **kwargs):
        if key == "adj":
            return torch.tensor([[self.xu.size(0)], [self.xv.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

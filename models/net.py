# Copyright 2021 Grabtaxi Holdings Pte Ltd (GRAB), All rights reserved.
# Use of this source code is governed by an MIT-style license that can be found in the LICENSE file

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import Tensor

from torch_sparse import SparseTensor
from torch_geometric.nn.dense.linear import Linear

from typing import Tuple, Union, Dict

from models.conv import BEANConv


def make_tuple(x: Union[int, Tuple[int, int, int], Tuple[int, int]], repeat: int = 3):
    if isinstance(x, int):
        if repeat == 2:
            return (x, x)
        else:
            return (x, x, x)
    else:
        return x


def apply_relu_dropout(x: Tensor, dropout_prob: float, training: bool) -> Tensor:
    x = F.relu(x)
    if dropout_prob > 0.0:
        x = F.dropout(x, p=dropout_prob, training=training)
    return x


class GraphBEAN(nn.Module):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int, int]],
        # in_channels_v: Union[int, Tuple[int, int, int]],
        # hidden_channels_u: Union[int, Tuple[int, int, int]] = 32,
        hidden_channels: Union[int, Tuple[int, int, int]] = 32,
        latent_channels: Union[int, Tuple[int, int]] = 64,
        # latent_channels_v: Union[int, Tuple[int, int]] = 64,
        edge_pred_latent: int = 64,
        node_pred_latent: int = 4,
        # node_pred_latent_v: int = 4,
        n_layers_encoder: int = 4,
        n_layers_decoder: int = 4,
        # n_layers_mlp: int = 4,
        n_layers_mlp_node: int = 4,
        n_layers_mlp_edge: int = 4,
        dropout_prob: float = 0.0,
    ):

        super().__init__()

        self.in_channels = make_tuple(in_channels)
        # self.in_channels_v = make_tuple(in_channels_v)
        # self.hidden_channels_u = make_tuple(hidden_channels_u)
        self.hidden_channels = make_tuple(hidden_channels)
        self.latent_channels = make_tuple(latent_channels, 2)
        # self.latent_channels_v = make_tuple(latent_channels_v, 2)
        self.edge_pred_latent = edge_pred_latent
        self.node_pred_latent = node_pred_latent
        # self.node_pred_latent_v = node_pred_latent_v
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        # self.n_layers_mlp = n_layers_mlp
        self.n_layers_mlp_node = n_layers_mlp_node
        self.n_layers_mlp_edge = n_layers_mlp_edge
        self.dropout_prob = dropout_prob

        self.create_encoder()
        self.create_class_decoder()
        self.create_feature_decoder()
        self.create_structure_decoder()

    def create_encoder(self):
        self.encoder_convs = nn.ModuleList()
        for i in range(self.n_layers_encoder):
            if i == 0:
                in_channels = self.in_channels
                out_channels = self.hidden_channels
            elif i == self.n_layers_encoder - 1:
                in_channels = self.hidden_channels
                out_channels = self.latent_channels
            else:
                in_channels = self.hidden_channels
                out_channels = self.hidden_channels

            if i == self.n_layers_encoder - 1:
                self.encoder_convs.append(
                    BEANConv(in_channels, out_channels, node_self_loop=False)
                )
            else:
                self.encoder_convs.append(
                    BEANConv(in_channels, out_channels, node_self_loop=True)
                )

    def create_feature_decoder(self):
        self.decoder_convs = nn.ModuleList()
        for i in range(self.n_layers_decoder):
            if i == 0:
                in_channels = self.latent_channels
                out_channels = self.hidden_channels
            elif i == self.n_layers_decoder - 1:
                in_channels = self.hidden_channels
                out_channels = self.in_channels
            else:
                in_channels = self.hidden_channels
                out_channels = self.hidden_channels

            self.decoder_convs.append(BEANConv(in_channels, out_channels))

    def create_class_decoder(self):
        layers = []
        input_dim = self.latent_channels[0]
        for _ in range(self.n_layers_mlp_node - 1):
            layers.append(nn.Linear(input_dim, self.node_pred_latent))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_prob))
            input_dim = self.node_pred_latent

        # Output layer
        layers.append(nn.Linear(self.node_pred_latent, 2))  # Assuming binary classification
        self.node_classifier = nn.Sequential(*layers)

    def create_structure_decoder(self):
        self.u_mlp_layers = nn.ModuleList()
        self.v_mlp_layers = nn.ModuleList()

        for i in range(self.n_layers_mlp_edge):
            if i == 0:
                in_channels = self.latent_channels
            else:
                in_channels = (self.edge_pred_latent, self.edge_pred_latent)
            out_channels = self.edge_pred_latent

            self.u_mlp_layers.append(Linear(in_channels[0], out_channels))
            self.v_mlp_layers.append(Linear(in_channels[1], out_channels))


    def forward(
        self,
        xu: Tensor,
        xv: Tensor,
        xe: Tensor,
        adj: SparseTensor,
        edge_pred_samples: SparseTensor,
    ) -> Dict[str, Tensor]:

        # encoder
        for i, conv in enumerate(self.encoder_convs):
            (xu, xv), xe = conv((xu, xv), adj, xe=xe)
            if i != self.n_layers_encoder - 1:
                xu = apply_relu_dropout(xu, self.dropout_prob, self.training)
                xv = apply_relu_dropout(xv, self.dropout_prob, self.training)
                xe = apply_relu_dropout(xe, self.dropout_prob, self.training)

        # feature decoder
        for i, conv in enumerate(self.decoder_convs):
            (xu, xv), xe = conv((xu, xv), adj, xe=xe)
            if i != self.n_layers_decoder - 1:
                xu = apply_relu_dropout(xu, self.dropout_prob, self.training)
                xv = apply_relu_dropout(xv, self.dropout_prob, self.training)
                xe = apply_relu_dropout(xe, self.dropout_prob, self.training)

        # get latent vars
        zu, zv = xu, xv

        # Class decoding:
        class_logits_u = self.node_classifier(zu)
        nprob_u = torch.sigmoid(class_logits_u)
        class_logits_v = self.node_classifier(zv)
        nprob_v = torch.sigmoid(class_logits_v)


        # structure decoder
        zu2, zv2 = zu, zv
        for i, layer in enumerate(self.u_mlp_layers):
            zu2 = layer(zu2)
            if i != self.n_layers_mlp_node - 1:
                zu2 = apply_relu_dropout(zu2, self.dropout_prob, self.training)

        for i, layer in enumerate(self.v_mlp_layers):
            zv2 = layer(zv2)
            if i != self.n_layers_mlp_node - 1:
                zv2 = apply_relu_dropout(zv2, self.dropout_prob, self.training)

        zu2_edge = zu2[edge_pred_samples.storage.row()]
        zv2_edge = zv2[edge_pred_samples.storage.col()]

        eprob = torch.sigmoid(torch.sum(zu2_edge * zv2_edge, dim=1))


        # collect results
        result = {"xu": xu, "xv": xv, "xe": xe, "zu": zu, "zv": zv, "eprob": eprob, "nprob_u": nprob_u, "nprob_v": nprob_v}

        return result

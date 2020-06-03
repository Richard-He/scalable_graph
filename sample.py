import os
import time
import argparse
import json
import math
import torch
import torch_geometric
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from argparse import Namespace
from torch_geometric.data import Data, Batch, NeighborSampler, ClusterData, ClusterLoader
import torch.nn as nn
import torch.distributed as dist
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tgcn import TGCN
from sandwich import Sandwich
from sampler import ImportanceSampler, NaiveSampler

from preprocess import generate_dataset, load_nyc_sharing_bike_data, load_metr_la_data, load_pems_d7_data, get_normalized_adj
from base_task import add_config_to_argparse, BaseConfig, BasePytorchTask, \
    LOSS_KEY, BAR_KEY, SCALAR_LOG_KEY, VAL_SCORE_KEY


class NeighborSampleDataset(IterableDataset):
    def __init__(self, X, y, edge_index, edge_weight, num_nodes, batch_size, graph_size, num_gcn_layer=1, shuffle=True,
                 use_dist_sampler=False, rep_eval=None, add_self_loop=False, cent_size=100, sample='ImportanceSampler'
                 ):
        self.X = X
        self.y = y
        self.add_self_loop = add_self_loop
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.shuffle = shuffle
        # whether to use distributed sampler when available
        self.use_dist_sampler = use_dist_sampler
        # use 'epoch' as the random seed to shuffle data for distributed training
        self.epoch = None

        # number of repeats to run evaluation, set to None for training mode
        self.rep_eval = rep_eval

        self.num_gcn_layer = num_gcn_layer
        self.sample = sample
        self.size = graph_size
        self.graph_sampler = self._make_graph_sampler()
        self.length = self.get_length()
        self.cent_size = cent_size

    def _make_graph_sampler(self):
        graph = Data(
            edge_index=self.edge_index, edge_attr=self.edge_weight, num_nodes=self.num_nodes
        ).to('cpu')

        if self.sample == 'ImportanceSampler':
            graph_sampler = ImportanceSampler(
                graph, size=np.repeat(self.size, self.num_gcn_layer), num_layers=self.num_gcn_layer,
                batch_size=self.cent_size,
                shuffle=self.shuffle, skip_connect=False, add_self_loop=self.add_self_loop
                # graph, size=[10, 15], num_hops=2, batch_size=250, shuffle=self.shuffle, add_self_loops=True
            )
        else:
            graph_sampler = NaiveSampler(
                graph, size=np.repeat(self.size, self.num_gcn_layer), num_layers=self.num_gcn_layer,
                batch_size=self.cent_size,
                shuffle=self.shuffle, skip_connect=False, add_self_loop=self.add_self_loop
                # graph, size=[10, 15], num_hops=2, batch_size=250, shuffle=self.shuffle, add_self_loops=True
            )

        return graph_sampler

    def get_subgraph(self, data_flow):
        sub_graph = {
            'edge_index': [block.edge_index for block in data_flow],
            'edge_weight': [self.edge_weight[block.e_id] for block in data_flow],
            'size': [block.size for block in data_flow],
            'cent_n_id': data_flow.n_id,
            'n_id': [block.n_id for block in data_flow],
            'graph_n_id': data_flow[0].n_id,
            'prob': [block.prob for block in data_flow],
            'skip_index': [skip_index for skip_index in data_flow.skip_edge_index]
        }
        return sub_graph

    def __iter__(self):
        repeats = 1 if self.rep_eval is None else self.rep_eval

        for rep in range(repeats):
            # decide random seeds for graph sampler

            if self.use_dist_sampler and dist.is_initialized():
                # ensure that all processes share the same graph dataflow
                # set seed as epoch for training, and rep for evaluation
                torch.manual_seed(self.epoch)

            if self.rep_eval is not None:
                # fix random seeds for repetitive evaluation
                # this attribute should not be set during training
                torch.manual_seed(rep)

            for data_flow in self.graph_sampler():
                g = self.get_subgraph(data_flow)
                X, y = self.X[:, g['graph_n_id']], self.y[:, g['cent_n_id']]
                dataset_len = X.size(0)
                indices = list(range(dataset_len))

                if self.use_dist_sampler and dist.is_initialized():
                    # distributed sampler reference: torch.utils.data.distributed.DistributedSampler
                    if self.shuffle:
                        # ensure that all processes share the same permutated indices
                        tg = torch.Generator()
                        tg.manual_seed(self.epoch)
                        indices = torch.randperm(
                            dataset_len, generator=tg).tolist()

                    world_size = dist.get_world_size()
                    node_rank = dist.get_rank()
                    num_samples_per_node = int(
                        math.ceil(dataset_len * 1.0 / world_size))
                    total_size = world_size * num_samples_per_node

                    # add extra samples to make it evenly divisible
                    indices += indices[:(total_size - dataset_len)]
                    assert len(indices) == total_size

                    # get sub-batch for each process
                    # Node (rank=x) get [x, x+world_size, x+2*world_size, ...]
                    indices = indices[node_rank:total_size:world_size]
                    assert len(indices) == num_samples_per_node
                elif self.shuffle:
                    np.random.shuffle(indices)

                num_batches = (len(indices) + self.batch_size -
                            1) // self.batch_size
                for batch_id in range(num_batches):
                    start = batch_id * self.batch_size
                    end = (batch_id + 1) * self.batch_size
                    yield X[indices[start: end]], y[indices[start: end]], g, torch.LongTensor(indices[start: end])

    def get_length(self):
        length = 0

        repeats = 1 if self.rep_eval is None else self.rep_eval

        for rep in range(repeats):
            for data_flow in self.graph_sampler():
                if self.use_dist_sampler and dist.is_initialized():
                    dataset_len = self.X.size(0)
                    world_size = dist.get_world_size()
                    num_samples_per_node = int(
                        math.ceil(dataset_len * 1.0 / world_size))
                else:
                    num_samples_per_node = self.X.size(0)
                length += (num_samples_per_node +
                        self.batch_size - 1) // self.batch_size

        return length

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        # self.set_epoch() will be called by BasePytorchTask on each epoch when using distributed training
        self.epoch = epoch
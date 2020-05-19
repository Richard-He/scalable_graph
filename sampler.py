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
import torch.nn.functional as f
import torch.distributed as dist
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from torch_geometric.data.sampler import Block, DataFlow
from torch_geometric.data import Data
from torch_geometric.utils import degree, segregate_self_loops
from torch_geometric.utils.repeat import repeat


class ImportanceSampler(object):
    """
    Importance Sampler implemented by Yihan.

    This sampler is different from the original version of neighbor sampler and incorporate a new workflow of
    importance sampling with skip-connection to it.

    Args:
        data (torch_geometric.data.Data): The graph data object.
        size(int or float or [int] or [float]): The number of nodes and precentages to sample( for each layer).
        batch_size: how many samples per batch
        shuffle : if set to 'True', the data will be shuffled
        drop_last : If set to 'True' will drop the last imcomplete batch
        flow(string) : The flow direction of message passing
        skip_connect : If set to 'True', will implement skip-connection
    """

    def __init__(self, data, size, num_layers, batch_size=1, shuffle=False, drop_last=False,
                 skip_connect=False, flow='source_to_target'):

        self.data = data
        self.size = repeat(size, num_layers)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.flow = flow
        self.skip_connect = skip_connect

        self.edge_index = data.edge_index
        self.edge_attr = data.edge_attr
        self.num_nodes = data.num_nodes

        A_weight = torch.sparse.FloatTensor(self.edge_index, self.edge_attr).to_dense()
        self.A_weight_normalize = f.normalize(A_weight, dim=0)

        self.e_id = torch.arange(self.edge_index.size(1))
        self.num_layers = num_layers

        assert flow in ['source_to_target', 'target_to_source']
        self.i, self.j = (0, 1) if flow == 'target_to_source' else (1, 0)

        edge_index_i, self.e_assoc = self.edge_index[self.i].sort()
        self.edge_index_j = self.edge_index[self.j, self.e_assoc]
        #deg = degree(edge_index_i, data.num_nodes, dtype=torch.long)
        #self.cumdeg = torch.cat([deg.new_zeros(1), deg.cumsum(0)])
        #cumdeg is the number of cumulative degree of nodes with index smaller than i

        self.e_ids = -1 * torch.ones([self.num_nodes, self.num_nodes], dtype=torch.long)
        for id in range(self.edge_index.size(1)):
            self.e_ids[self.edge_index[self.i][id], self.edge_index[self.j][id]] = id

        self.tmp = torch.empty(data.num_nodes, dtype=torch.long)


    def __get_batches__(self, subset=None):
        if subset is None and not self.shuffle:
            subset = torch.arange(self.num_nodes, dtype=torch.long)
        elif subset is None and self.shuffle:
            subset = torch.randperm(self.num_nodes)
        else:
            if subset.dtype == torch.bool or subset.dtype == torch.uint8:
                subset = subset.nonzero().view(-1)
            if self.shuffle:
                subset = subset[torch.randperm(subset.size(0))]

        subsets = torch.split(subset, self.batch_size)
        if self.drop_last and subsets[-1].size(0) < self.batch_size:
            subsets = subsets[:-1]
        assert len(subsets) > 0
        return subsets


    def get_distribution(self, ids):
        """
        ids: torch.Tensor of ids to extract distribution of next layer's id
        format:(0, 1, 0, 1...)
        getting the neighbor_node distribution of node #id
        """
        idmask = torch.zeros(self.num_nodes)
        for id in ids:
            idmask[id] = 1
        dist = torch.matmul(self.A_weight_normalize, idmask)
        dist = dist/dist.sum(0).expand_as(dist)
        return dist


    def sampler(self, n_id, size):
        """
        sampler for datasets
        returning new n_id and e_id in edge_index
        """
        dist = self.get_distribution(n_id)
        ndist = dist.numpy()
        maxsz = np.count_nonzero(ndist)

        if maxsz < size:
            size = maxsz
        samp = np.random.choice(range(self.num_nodes), size=size, replace=False, p=ndist)
        samp, _ = torch.from_numpy(samp).sort()
        e_id = []
        for sam in samp:
            for id in n_id:
                if self.e_ids[sam, id] + 1 != 0:
                    e_id.append(self.e_ids[sam, id])
        return samp, e_id


    def __produce_block__(self, n_id_i, n_id_j):
        e_id = []
        for idj in n_id_j:
            for idi in n_id_i:
                if self.e_ids[idj, idi] + 1 != 0:
                    e_id.append(self.e_ids[idj, idi])

        edge_index_i = self.edge_index[self.i, e_id]
        self.tmp[n_id] = torch.arange(n_id.size(0))
        edges[self.i] = self.tmp[edge_index_i]

        edge_index_j = self.edge_index[self.j, e_id]
        self.tmp[n_id_j] = torch.arange(n_id_j.size(0))
        edges[self.j] = self.tmp[edge_index_j]

        edge_index = torch.stack(edges, dim=0)
        return (n_id_j, None, e_id, edge_index), n_id_i

    def __produce_bipartite_data_flow_importance__(self, n_id):
        data_flow = DataFlow(n_id, self.flow)
        for l in range(self.num_layers):

            #prepare for edge_index in the block
            edges = [None, None]
            new_n_id, e_id = self.sampler(n_id, self.size[l])
            e_id = self.e_id[e_id]

            edge_index_i = self.edge_index[self.i, e_id]
            self.tmp[n_id] = torch.arange(n_id.size(0))
            edges[self.i] = self.tmp[edge_index_i]

            edge_index_j = self.edge_index[self.j, e_id]
            self.tmp[new_n_id] = torch.arange(new_n_id.size(0))
            edges[self.j] = self.tmp[edge_index_j]

            edge_index = torch.stack(edges, dim=0)
            n_id = new_n_id
            data_flow.append(n_id, None, e_id, edge_index)

        return data_flow


    # def __add_residual__(self, dataflow):
    #     n_id = dataflow.n_id
    #     for i in range()



    def __call__(self, subset=None):
        produce = self.__produce_bipartite_data_flow_importance__
        for n_id in self.__get_batches__(subset):
            yield produce(n_id)

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

from torch_geometric.data import Data
from torch_geometric.utils import degree, segregate_self_loops, remove_isolated_nodes
from torch_geometric.utils.repeat import repeat


class Block(object):
    def __init__(self, n_id, res_n_id, e_id, edge_index, size, prob):
        self.n_id = n_id
        self.res_n_id = res_n_id
        self.e_id = e_id
        self.edge_index = edge_index
        self.size = size
        self.prob = prob

    def __repr__(self):
        info = [(key, getattr(self, key)) for key in self.__dict__]
        info = ['{}={}'.format(key, size_repr(item)) for key, item in info]
        return '{}({})'.format(self.__class__.__name__, ', '.join(info))


class DataFlow(object):
    def __init__(self, n_id, flow='source_to_target'):
        self.n_id = n_id
        self.flow = flow
        self.__last_n_id__ = n_id
        self.blocks = []
        self.skip_edge_index = []

    @property
    def batch_size(self):
        return self.n_id.size(0)

    def append(self, n_id, res_n_id, e_id, edge_index, distribution):
        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        size = [None, None]
        size[i] = self.__last_n_id__.size(0)
        size[j] = n_id.size(0)
        block = Block(n_id, res_n_id, e_id, edge_index, tuple(size), distribution)
        self.blocks.append(block)
        self.__last_n_id__ = n_id

    def add_skip_index(self, edge_index):
        self.skip_edge_index = edge_index

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[::-1][idx]

    def __iter__(self):
        for block in self.blocks[::-1]:
            yield block

    def to(self, device):
        for block in self.blocks:
            block.edge_index = block.edge_index.to(device)
        return self

    def __repr__(self):
        n_ids = [self.n_id] + [block.n_id for block in self.blocks]
        sep = '<-' if self.flow == 'source_to_target' else '->'
        info = sep.join([str(n_id.size(0)) for n_id in n_ids])
        return '{}({})'.format(self.__class__.__name__, info)


class NaiveSampler(object):
    def __init__(self, data, size, num_layers, batch_size=1, shuffle=False, drop_last=False,
                 flow='source_to_target', add_self_loop=True, skip_connect=False):

        self.data = data
        self.size = repeat(size, num_layers)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.flow = flow
        self.skip_connect = skip_connect
        self.self_loop = add_self_loop
        self.edge_index = data.edge_index
        self.edge_attr = data.edge_attr
        self.num_nodes = data.num_nodes

        A_weight = torch.sparse.FloatTensor(self.edge_index, self.edge_attr).to_dense()
        self.A_weight_normalize = f.normalize(A_weight, p=1, dim=1)

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
            self.e_ids[self.edge_index[self.j][id], self.edge_index[self.i][id]] = id

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


    def __produce_bipartite_data_flow_importance__(self, n_id):
        n_id, _ = n_id.sort()
        data_flow = DataFlow(n_id, self.flow)

        data_flow.append(n_id, None, None, None, None)
        return data_flow


    def __call__(self, subset=None):
        produce = self.__produce_bipartite_data_flow_importance__
        for n_id in self.__get_batches__(subset):
            yield produce(n_id)



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
                 flow='source_to_target', add_self_loop=True, skip_connect=False):

        self.data = data
        self.size = repeat(size, num_layers)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.flow = flow
        self.skip_connect = skip_connect
        self.self_loop = add_self_loop
        self.edge_index = data.edge_index
        self.edge_attr = data.edge_attr
        self.num_nodes = data.num_nodes

        A_weight = torch.sparse.FloatTensor(self.edge_index, self.edge_attr).to_dense()
        self.A_weight_normalize = f.normalize(A_weight, p=1, dim=1)

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
            self.e_ids[self.edge_index[self.j][id], self.edge_index[self.i][id]] = id

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
        idmask[ids] = 1
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
        samp = torch.from_numpy(samp)
        e_id = []

        for sam in samp:
            for id in n_id:
                if self.e_ids[sam, id] + 1 != 0:
                    e_id.append(self.e_ids[sam, id])

        con_id = self.edge_index[self.i, e_id]
        # find the isolated nodes
        if self.self_loop:
            samp = torch.cat([samp, n_id]).unique()
            for node in n_id:
                e_id.append(self.e_ids[node, node])
            e_id = list(set(e_id))
        else:
            isolate_nodes = torch.LongTensor(list(set(n_id.tolist()) - set(con_id.tolist())))
            samp = torch.cat([samp, isolate_nodes])
            for node in isolate_nodes:
                e_id.append(self.e_ids[node, node])
        samp, _ = samp.sort()
        p = torch.from_numpy(ndist)[samp]
        return samp, e_id, p


    def produce_edge_index(self, n_id_j, n_id_i):
        e_id = []
        for j in n_id_j:
            for i in n_id_i:
                if self.e_ids[j, i] + 1 != 0:
                    e_id.append(self.e_ids[j, i])

        edges = [None, None]
        # produce mapping
        edge_index_i = self.edge_index[self.i, e_id]
        self.tmp[n_id_i] = torch.arange(n_id_i.size(0))
        edges[self.i] = self.tmp[edge_index_i]
        edge_index_j = self.edge_index[self.j, e_id]
        self.tmp[n_id_j] = torch.arange(n_id_j.size(0))
        edges[self.j] = self.tmp[edge_index_j]

        edge_index = torch.stack(edges, dim=0)
        return edge_index


    def find_residual(self, data_flow):
        edge_indexes = []

        for i in range(len(data_flow)-2):
            n_id_j = data_flow[i].n_id
            n_id_i = data_flow[i+2].n_id
            edge_index = self.produce_edge_index(n_id_j, n_id_i)
            edge_indexes.append(edge_index)
        return edge_indexes


    def __produce_bipartite_data_flow_importance__(self, n_id):
        n_id, _ = n_id.sort()
        data_flow = DataFlow(n_id, self.flow)
        for l in range(self.num_layers):

            #prepare for edge_index in the block
            edges = [None, None]
            new_n_id, e_id, p = self.sampler(n_id, self.size[l])

            #new_n_id is sorted
            e_id = self.e_id[e_id]

            #mapping real edge_index to flow edge_index
            edge_index_i = self.edge_index[self.i, e_id]
            self.tmp[n_id] = torch.arange(n_id.size(0))
            edges[self.i] = self.tmp[edge_index_i]
            edge_index_j = self.edge_index[self.j, e_id]
            self.tmp[new_n_id] = torch.arange(new_n_id.size(0))
            edges[self.j] = self.tmp[edge_index_j]

            #create new edge_index
            edge_index = torch.stack(edges, dim=0)

            assert len(edge_index[self.i].unique()) == len(n_id)
            assert len(edge_index[self.j].unique()) == len(new_n_id)
            #update new_n_id
            n_id = new_n_id
            data_flow.append(n_id, None, e_id, edge_index, p)


        if self.skip_connect:
            edge_index = find_residual(data_flow)
            data_flow.add_skip_index(edge_index)

        return data_flow


    # def __add_residual__(self, dataflow):
    #     n_id = dataflow.n_id
    #     for i in range()



    def __call__(self, subset=None):
        produce = self.__produce_bipartite_data_flow_importance__
        for n_id in self.__get_batches__(subset):
            yield produce(n_id)

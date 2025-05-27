import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from conv import GNN_node, GNN_node_Virtualnode


class SimpleGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGCN, self).__init__()
        self.embedding = torch.nn.Embedding(1, input_dim) 
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.global_pool = global_mean_pool  
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x)  
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = self.global_pool(x, batch)  
        out = self.fc(x)  
        return out



class GNN(torch.nn.Module):

    def __init__(self, num_class, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_class)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_class)

    def forward(self, batched_data, return_all=False):
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch

        if self.training:
            # Add noise to node features (random normal noise scaled by 0.4)
            noise = torch.randn_like(x) * 0.4
            x_noisy = x + noise

            # Drop edges randomly with drop probability 0.2
            edge_index_perturbed = self.drop_edges(edge_index, drop_prob=0.2)

            # First pass: noisy node features + original edges
            batched_data.x = x_noisy
            batched_data.edge_index = edge_index
            h_node = self.gnn_node(batched_data)
            h_graph = self.pool(h_node, batch)
            out = self.graph_pred_linear(h_graph)

            # Second pass: same noisy features + perturbed edges
            batched_data.x = x_noisy  # reuse same noisy features
            batched_data.edge_index = edge_index_perturbed
            h_node_perturbed = self.gnn_node(batched_data)
            h_graph_perturbed = self.pool(h_node_perturbed, batch)
            out_perturbed = self.graph_pred_linear(h_graph_perturbed)

            if return_all:
                return out, out_perturbed
            else:
                return out

        else:
            # Evaluation mode: no noise, no edge dropout
            batched_data.x = x
            batched_data.edge_index = edge_index
            h_node = self.gnn_node(batched_data)
            h_graph = self.pool(h_node, batch)
            out = self.graph_pred_linear(h_graph)
            return out
    
    def drop_edges(self, edge_index, drop_prob):
        # Get mask for keeping edges
        mask = torch.rand(edge_index.size(1), device=edge_index.device) >= drop_prob
        return edge_index[:, mask]
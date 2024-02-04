from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATv2Conv, GINConv, LayerNorm, ChebConv, BatchNorm, GATConv, AGNNConv, GatedGraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from transformers import AutoModel
from torch_geometric.nn import aggr
from torch.utils import checkpoint


class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, drop, conv_layer, use_checkpointing= False):
        super(GraphEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.drop = drop
        self.relu = nn.ReLU()
        
        # self.use_checkpointing = use_checkpointing
        self.dropout = nn.Dropout(p=0.2)
        
        self.lnorm = nn.LayerNorm((nout))
        # self.lnorm = LayerNorm((graph_hidden_channels))
        # self.softmax_aggr = aggr.SoftmaxAggregation(learn=True)

        
        if conv_layer == 'GCN':
            self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
            self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
            self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
            
        elif conv_layer == 'GAT':
            self.conv1 = GATv2Conv(num_node_features, graph_hidden_channels)
            self.conv2 = GATv2Conv(graph_hidden_channels, graph_hidden_channels)
            self.conv3 = GATv2Conv(graph_hidden_channels, graph_hidden_channels)
            
        elif conv_layer == 'GIN':

            self.conv1 = GINConv(nn.Linear(num_node_features, graph_hidden_channels))
            self.conv2 = GINConv(nn.Linear(graph_hidden_channels, graph_hidden_channels))
            self.conv3 = GINConv(nn.Linear(graph_hidden_channels, graph_hidden_channels))
        
        elif conv_layer == 'AGNN':
            self.conv1 = AGNNConv(num_node_features, graph_hidden_channels)
            self.conv2 = AGNNConv(graph_hidden_channels, graph_hidden_channels)
            self.conv3 = AGNNConv(graph_hidden_channels, graph_hidden_channels)
        
        elif conv_layer == 'GatedGraphConv':        
            self.conv1 = GatedGraphConv(num_node_features, graph_hidden_channels)
            self.conv2 = GatedGraphConv(graph_hidden_channels, graph_hidden_channels)
            self.conv3 = GatedGraphConv(graph_hidden_channels, graph_hidden_channels)
            
        elif conv_layer == 'ChebConv':
            self.conv1 = ChebConv(num_node_features, graph_hidden_channels, 4)
            self.conv2 = ChebConv(graph_hidden_channels, graph_hidden_channels, 4)
            self.conv3 = ChebConv(graph_hidden_channels, graph_hidden_channels, 4)
        
        self.bn1 = BatchNorm(graph_hidden_channels)
        self.bn2 = BatchNorm(graph_hidden_channels)
        self.bn3 = BatchNorm(graph_hidden_channels)
        
        self.attention = GATConv(graph_hidden_channels, graph_hidden_channels, heads=1)
        self.bn_attention = BatchNorm(graph_hidden_channels)

        self.lin1 = nn.Linear(graph_hidden_channels, nhid)
        self.lin2 = nn.Linear(nhid, nhid)
        self.lin3 = nn.Linear(nhid, nhid)
        self.lin4 = nn.Linear(nhid, nout)
     

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        
        x = self.relu(self.bn1(self.conv1(x, edge_index))) + x 
        x = self.relu(self.bn2(self.conv2(x, edge_index))) + x
        x = self.relu(self.bn3(self.conv3(x, edge_index))) + x
        x = self.relu(self.bn_attention(self.attention(x, edge_index))) 
        
        x = global_add_pool(x, batch)
        # x = global_mean_pool(x, batch)
        # x = global_max_pool(x,batch)
        
        if self.drop:
            x = self.relu(self.lin1(x))
            # x = self.dropout(x)
            x = self.relu(self.lin2(x))
            x = self.dropout(x)
            x = self.relu(self.lin3(x))
            x = self.dropout(x)
            x = self.lnorm(self.lin4(x))
        else:
            x = self.relu(self.lin1(x))
            x = self.relu(self.lin2(x))
            x = self.relu(self.lin3(x))
            x = self.lnorm(self.lin4(x))
        return x
    
class TextEncoder(nn.Module):
    def __init__(self, model_name, nout):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, nout)
        self.norm = nn.LayerNorm(nout)
        
    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        cls_token_state = encoded_text.last_hidden_state[:, 0, :]
        linear_output = self.linear(cls_token_state)
        normalized_output = self.norm(linear_output)
        return normalized_output
    
class Model(nn.Module):
    def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels, drop, conv_layer, use_checkpointing = False):
        super(Model, self).__init__()
        self.graph_encoder = GraphEncoder(num_node_features, nout, nhid, graph_hidden_channels, drop, conv_layer,  use_checkpointing=use_checkpointing)
        self.text_encoder = TextEncoder(model_name, nout)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder

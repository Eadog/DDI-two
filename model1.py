import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch_geometric.nn import Set2Set

from Layer import TransformerEncoderLayer

from torch_scatter import scatter_mean, scatter_add, scatter_std
import time
import torch_geometric.nn as gnn
from mambapy.mamba import Mamba, MambaConfig



class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(self, x, edge_index, complete_edge_index,
            subgraph_node_index=None, subgraph_edge_index=None,
            subgraph_edge_attr=None, subgraph_indicator_index=None, 
            edge_attr=None, degree=None, ptr=None, return_attn=False):
        output = x

        for mod in self.layers:
            output = mod(output, edge_index, complete_edge_index,
                edge_attr=edge_attr, degree=degree,
                subgraph_node_index=subgraph_node_index,
                subgraph_edge_index=subgraph_edge_index,
                subgraph_indicator_index=subgraph_indicator_index, 
                subgraph_edge_attr=subgraph_edge_attr,
                ptr=ptr,
                return_attn=return_attn
            )
        if self.norm is not None:
            output = self.norm(output)
        return output



class D1Model1(nn.Module):
    def __init__(self, config, 
                node_input_dim=133,
                edge_input_dim=14,
                node_hidden_dim=300,
                edge_hidden_dim=300,
                num_step_message_passing=3,
                interaction='dot',
                num_prototypes=4,
                latent_dim=896,
                num_heads=8, dim_feedforward=512, dropout=0.0, num_layers=4, dim_hidden=64,
                batch_norm=False, abs_pe=False, abs_pe_dim=0,
                gnn_type="graph", se="gnn", use_edge_attr=False, num_edge_features=4,
                in_embed=True, edge_embed=True, use_global_pool=True, max_seq_len=None,
                global_pool='mean',use_subgraph=True, **kwargs
                ):
        
        super(D1Model1, self).__init__()

        self.config = config
        self.node_input_dim = node_input_dim
        self.node_hidden_dim = node_hidden_dim
        self.edge_input_dim = edge_input_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.num_step_message_passing = num_step_message_passing
        self.interaction = interaction
        self.latent_dim = latent_dim
        self.init_model()
        #zhang
        self.best_acc = -1.0
        self.best_roc_auc = -1.0
        self.best_f1 = -1.0
        self.best_ap = -1.0
        self.map_s = nn.Linear(384, 64)
        
       
        self.gamma_raw = nn.Parameter(torch.tensor(0.0))  # 初始值sigmoid为0.5
        self.activation = nn.ReLU()

################################图处理部分
        self.use_subgraph = use_subgraph
        self.use_edge_attr = use_edge_attr
        self.atom_embedding = nn.Linear(in_features=node_input_dim, out_features=dim_hidden, bias=False)
        
        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            edge_dim = kwargs.get('edge_dim', 32)
            self.embedding_edge = nn.Linear(in_features=num_edge_features, out_features=edge_dim, bias=False)
        else:
            kwargs['edge_dim'] = None
        
        encoder_layer = TransformerEncoderLayer(dim_hidden, num_heads, dim_feedforward, dropout, batch_norm=batch_norm, **kwargs)
        self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)

        # 全局池化（计算图的整体特征）
        self.global_pool = global_pool
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool


        # Classification layer
        #三分类
        # self.mlp = nn.Sequential(
            
        #     nn.Linear(4803, 1024),
        #     #nn.Linear(2403, 1024),
        #     nn.BatchNorm1d(1024), 
        #     nn.ReLU(),
        #     nn.Dropout(config.dropout),
 
        #     nn.Linear(1024, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Dropout(config.dropout),
            
        #     nn.Linear(512, 3)
        # )
        
        #二分类
        self.mlp2 = nn.Sequential(
            
            # nn.Linear(1664, 512),
            # #nn.Linear(2403, 1024),
            # nn.BatchNorm1d(512), 
            # nn.ReLU(),
            # nn.Dropout(config.dropout),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(64, 1)
        )
        # 消融最终融合模块
        # self.mlp2 = nn.Sequential(
            
        #     # nn.Linear(1664, 512),
        #     # #nn.Linear(2403, 1024),
        #     # nn.BatchNorm1d(512), 
        #     # nn.ReLU(),
        #     # nn.Dropout(config.dropout),

        #     nn.Linear(256, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Dropout(config.dropout),
            
        #     nn.Linear(64, 1)
        # )
        #消融序列模块
        self.mlp2 = nn.Sequential(
            
            # nn.Linear(1664, 512),
            # #nn.Linear(2403, 1024),
            # nn.BatchNorm1d(512), 
            # nn.ReLU(),
            # nn.Dropout(config.dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            
            nn.Linear(64, 1)
        )

        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def init_model(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.fill_(0.0)
    def cosine_similarity(self, x1, x2):
        cos_sim = F.cosine_similarity(x1, x2, dim=1)
        return cos_sim

        

    def forward(self, data1, data2, return_attn=False):
       
        smiles_1 = data1.xulie #[1013,512]
        smiles_2 = data2.xulie
        x1, edge_index1, edge_attr1 = data1.x, data1.edge_index, data1.edge_attr
        x2, edge_index2, edge_attr2 = data2.x, data2.edge_index, data2.edge_attr
        smiles_1 = self.map_s(smiles_1)
        smiles_2 = self.map_s(smiles_2)
        
        
        #s1、s2是序列特征
        
        ##以下是图处理部分
        complete_edge_index1 = data1.complete_edge_index if hasattr(data1, 'complete_edge_index') else None
        complete_edge_index2 = data2.complete_edge_index if hasattr(data2, 'complete_edge_index') else None
        graph_feature1 = self.atom_embedding(x1)
        
        if self.use_edge_attr and edge_attr1 is not None:
            edge_attr1 = self.embedding_edge(edge_attr1)
        
        else:
            edge_attr1 = None
      
        if self.use_subgraph:
            subgraph_node_index1 = data1.subgraph_node_idx if hasattr(data1, "subgraph_node_idx") else None
            subgraph_edge_index1 = data1.subgraph_edge_index if hasattr(data1, "subgraph_edge_index") else None
            subgraph_indicator_index1 = data1.subgraph_indicator if hasattr(data1, "subgraph_indicator") else None
            subgraph_edge_attr1 = data1.subgraph_edge_attr if hasattr(data1, "subgraph_edge_attr") else None
        else:
            subgraph_node_index1 = None
            subgraph_edge_index1 = None
            subgraph_indicator_index1 = None
            subgraph_edge_attr1 = None
        # Transformer 编码器
        graph_feature1 = self.encoder(
            graph_feature1, 
            edge_index1, 
            complete_edge_index=complete_edge_index1,
            edge_attr=edge_attr1, 
            subgraph_node_index=subgraph_node_index1,
            subgraph_edge_index=subgraph_edge_index1,
            subgraph_indicator_index=subgraph_indicator_index1, 
            subgraph_edge_attr=subgraph_edge_attr1,
            ptr=data1.ptr,
            return_attn=return_attn
        )
        # 读取整个分子的特征
        graph_feature1 = self.pooling(graph_feature1, data1.batch)

        graph_feature2 = self.atom_embedding(x2)
        if self.use_edge_attr and edge_attr2 is not None:
            edge_attr2 = self.embedding_edge(edge_attr2)
        else:
            edge_attr2 = None
      
        if self.use_subgraph:
            subgraph_node_index2 = data2.subgraph_node_idx if hasattr(data2, "subgraph_node_idx") else None
            subgraph_edge_index2 = data2.subgraph_edge_index if hasattr(data2, "subgraph_edge_index") else None
            subgraph_indicator_index2 = data2.subgraph_indicator if hasattr(data2, "subgraph_indicator") else None
            subgraph_edge_attr2 = data2.subgraph_edge_attr if hasattr(data2, "subgraph_edge_attr") else None
        else:
            subgraph_node_index2 = None
            subgraph_edge_index2 = None
            subgraph_indicator_index2 = None
            subgraph_edge_attr2 = None
        # Transformer 编码器
        graph_feature2 = self.encoder(
            graph_feature2, 
            edge_index2, 
            complete_edge_index=complete_edge_index2,
            edge_attr=edge_attr2, 
            subgraph_node_index=subgraph_node_index2,
            subgraph_edge_index=subgraph_edge_index2,
            subgraph_indicator_index=subgraph_indicator_index2, 
            subgraph_edge_attr=subgraph_edge_attr2,
            ptr=data2.ptr,
            return_attn=return_attn
        )
        # 读取整个分子的特征
        graph_feature2 = self.pooling(graph_feature2, data2.batch)

        #维度：graph_feature1:512,64  x
        gamma = torch.sigmoid(self.gamma_raw)
        fused1 = (1 - gamma) * graph_feature1 + gamma * smiles_1
        fused2 = (1 - gamma) * graph_feature2 + gamma * smiles_2
        hidden_states = torch.cat((fused1, fused2), dim=1)
        # 4. 特征融合
        #hidden_states = torch.cat((graph_feature1, graph_feature2), dim=1) #512,4800
       
        # 5. 分类层
        #三分类
        # logits = self.mlp(final_embedding)
        #二分类
        logits = self.mlp2(hidden_states) 
        
        return logits
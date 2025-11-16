"""
Code for the paper in MICCAI 2025:
"Anatomical Graph-based Multilevel Distillation for Robust Alzheimer's Disease Diagnosis with Missing Modalities"
URL: https://papers.miccai.org/miccai-2025/paper/3438_paper.pdf
"""
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torch_geometric.nn import GCNConv, TransformerConv
import numpy as np
from AGMD import utils

warnings.filterwarnings("ignore")


class Result(object):
    def __init__(self, d: dict):
        for k in d.keys():
            setattr(self, k, d[k])


class Classifier(torch.nn.Module):
    def __init__(self, num_features=512, mid_features=64, num_class=2):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(num_features, mid_features)
        self.fc2 = nn.Linear(mid_features, num_class)
        self.relu = nn.ReLU()

    def forward(self, data, l=None, training=True):
        x = data.view(data.size(0), -1)
        x1 = self.fc1(x)
        x = self.relu(x1)
        x = self.fc2(x)
        if training:
            log_out = F.log_softmax(x, dim=1)
            p_log_out = F.softmax(x, dim=1)
            loss = F.nll_loss(log_out, l)
            return Result({"fc": x1, "logit": x, "y": log_out, "p": p_log_out, "loss": loss})
        else:
            return F.softmax(x, dim=1)


class vgg16(nn.Module):
    def __init__(self, **kwargs):
        super(vgg16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, dilation=1),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, dilation=1),
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2, dilation=1),
            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]


class Encoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = vgg16()  # it is a default encoder, u can use your encoder

    def forward(self, x):
        return self.encoder(x)


class CrossModalAttention3D(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Conv3d(embed_dim, embed_dim, 1)
        self.key = nn.Conv3d(embed_dim, embed_dim, 1)
        self.value = nn.Conv3d(embed_dim, embed_dim, 1)

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(embed_dim, embed_dim // 2, 1),
            nn.ReLU(),
            nn.Conv3d(embed_dim // 2, embed_dim, 1),
            nn.Sigmoid()
        )

        self.conv1 = nn.Sequential(nn.Conv3d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm3d(embed_dim // 2),
                                   nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=1))

    def forward(self, mri_feat, pet_feat):
        channel_weight = self.channel_attn(mri_feat + pet_feat)
        mri_feat = mri_feat * channel_weight
        pet_feat = pet_feat * channel_weight

        Q = self.query(mri_feat)
        K = self.key(pet_feat)
        V = self.value(pet_feat)

        attn_map = F.softmax(torch.einsum('bcdhw,bcDHW->bdhwDHW', Q, K), dim=-1)
        fused_feat = torch.einsum('bdhwDHW,bcDHW->bcdhw', attn_map, V)

        return self.conv1(fused_feat + mri_feat)


class CrossModalFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(embed_dim, embed_dim // 2, 1),
            nn.ReLU(),
            nn.Conv3d(embed_dim // 2, embed_dim, 1),
            nn.Sigmoid()
        )

        self.conv1 = nn.Sequential(nn.Conv3d(embed_dim * 2, embed_dim // 2, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm3d(embed_dim // 2),
                                   nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=1))

    def forward(self, mri_feat, pet_feat):
        channel_weight = self.channel_attn(mri_feat + pet_feat)
        mri_feat = mri_feat * channel_weight
        pet_feat = pet_feat * channel_weight
        fused = torch.cat([mri_feat, pet_feat], dim=1)
        return self.conv1(fused)


class GraphBuilder3D:
    templates = None
    private_nodes = None

    @staticmethod
    def init(cfg=None):
        templates, private_nodes = utils.get_template(cfg.atlas_path)
        GraphBuilder3D.templates = templates
        GraphBuilder3D.private_nodes = private_nodes

    @staticmethod
    def build_graph(extracted_features, layer, label, cfg=None):
        """
        input: features [b, c, d, h, w]
        output: List of Data(one subject one graph)
        """
        feature_map = extracted_features[layer]
        template, private_node = GraphBuilder3D.templates[layer], GraphBuilder3D.private_nodes[layer]

        node_feature = utils.get_roi_feature(feature_map, template, private_node)
        node_feature = torch.stack(node_feature, dim=0).transpose(0, 1)

        if cfg and cfg.anatomical_graph:
            # build anatomical graph
            anatomical_edges = utils.add_anatomical_edges(private_node)
            graphs = utils.build_graph_structure_batch(node_feature, label, math.ceil(math.sqrt(len(node_feature))),
                                                       anatomical_edges)
        else:  # cosine_similarity + KNN
            graphs = utils.build_graph_structure_batch(node_feature, label)

        return graphs


class GCNModule(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.drop_edge = 0.5

    def _drop_edges(self, edge_index, edge_weights):
        if self.training and self.drop_edge > 0:
            mask = torch.rand(edge_index.size(1)) > self.drop_edge
            edge_index = edge_index[:, mask]
            edge_weights = edge_weights[mask]
        return edge_index, edge_weights

    def forward(self, graphs):
        return [self._forward_single(g) for g in graphs]

    def _forward_single(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = self.conv2(x, data.edge_index)
        return x


class Graph_Transformer(nn.Module):
    def __init__(self, input_dim, head_num=4, hidden_dim=64):
        super(Graph_Transformer, self).__init__()
        self.graph_conv = TransformerConv(input_dim, input_dim // head_num, head_num)
        self.lin_out = nn.Linear(input_dim, input_dim)

        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, input_dim)
        self.act = nn.ReLU()

    def forward(self, x, edge_index):
        out1 = self.lin_out(self.graph_conv(x, edge_index))
        out2 = self.ln1(out1 + x)
        out3 = self.lin2(self.act(self.lin1(out2)))
        out4 = self.ln2(out3 + out2)

        return out4


class TransformerFusion(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=4):
        super(TransformerFusion, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.down = nn.Sequential(nn.Linear(embed_dim, 512), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.down(self.transformer_encoder(x))


class TeacherModel3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mri_encoder = Encoder3D()
        self.pet_encoder = Encoder3D()
        # you will not want to use the first two layers!
        if cfg.CMT:
            self.cross_attns = nn.ModuleList([
                None,  # CrossModalAttention3D(64),   # shallow features
                None,  # CrossModalAttention3D(128),  # middle features
                CrossModalAttention3D(256),  # middle features
                CrossModalAttention3D(512)  # deep features
            ])
        else:
            self.cross_attns = nn.ModuleList([
                None,  # CrossModalFusion(64),   # shallow features
                None,  # CrossModalFusion(128),  # middle features
                CrossModalFusion(256),  # middle features
                CrossModalFusion(512)  # deep features
            ])
        self.gcn = torch.nn.ModuleList([GCNModule(256), GCNModule(512)])  # only take last two layers

        self.transformer = TransformerFusion(100 * 64, 8)
        self.classifier = Classifier(512, num_class=cfg.num_classes)

    def forward(self, mri, pet, label):
        # get multi-layer features
        mri_features = self.mri_encoder(mri)
        pet_features = self.pet_encoder(pet)

        # fusing features across layers between modalities
        fused_features = []
        for i in range(len(self.cross_attns)):
            if self.cross_attns[i] is None:
                continue
            fused = self.cross_attns[i](mri_features[i], pet_features[i])
            fused_features.append(fused)

        # graph process
        sub_graphs = []  # graphs in each layer
        for layer in (-1, -2):
            # build graph from fused features
            graphs = GraphBuilder3D.build_graph(fused_features, layer=layer, label=None, cfg=self.cfg)
            gcn_out = self.gcn[layer](graphs)
            graph = torch.stack(gcn_out, dim=0)
            sub_graphs.append(graph)

        # Transformer
        trans_input = torch.cat(sub_graphs, dim=1)
        global_feat = self.transformer(trans_input.view(trans_input.size(0), -1))
        out = self.classifier(global_feat, label)

        return {
            'mid': fused_features[0],
            'deep': fused_features[1],
            'graph': trans_input,
            'global': global_feat.squeeze(1),
            'cls': out
        }


class StudentModel3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder3D()
        self.gcn = torch.nn.ModuleList([GCNModule(256), GCNModule(512)])
        self.transformer = TransformerFusion(100 * 64, 8, num_layers=4)  # without remove same ones
        self.classifier = Classifier(512, num_class=cfg.num_classes)
        self.cfg = cfg

    def forward(self, mri, label=None):
        features = self.encoder(mri)

        # graph process
        sub_graphs = []
        for layer in (-1, -2):
            graphs = GraphBuilder3D.build_graph(features, layer=layer, label=None)
            gcn_out = self.gcn[layer](graphs)
            graph = torch.stack(gcn_out, dim=0)
            sub_graphs.append(graph)

        # Transformer
        trans_input = torch.cat(sub_graphs, dim=1)
        global_feat = self.transformer(trans_input.view(trans_input.size(0), -1))
        out = self.classifier(global_feat, label, self.cfg.training)

        return {
            'mid': features[2],
            'deep': features[3],
            'graph': trans_input,
            'global': global_feat.squeeze(1),
            'cls': out
        }


class UncertaintyWeightedDistiller:
    def __init__(self, temp=0.5):
        self.temp = temp

    def calc_weights(self, teacher_output):
        weights = {}
        for level in ['mid', 'deep']:
            feat = teacher_output[level]
            prob = F.softmax(feat.mean(dim=[2, 3, 4]), dim=-1)
            entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
            weights[level] = 1 - entropy / np.log(teacher_output[level].size(1))

        global_prob = F.softmax(teacher_output['global'], dim=-1)
        global_entropy = -torch.sum(global_prob * torch.log(global_prob + 1e-10), dim=1)
        weights['global'] = 1 - global_entropy / np.log(teacher_output['global'].size(1))

        return weights


class DiscriminatorDistiller(nn.Module):
    def __init__(self, encoder_fet=True, in_dim=512, cfg=None):
        super(DiscriminatorDistiller).__init__()
        self.is_feat_from_encoder = encoder_fet
        self.in_dim = in_dim
        if self.is_feat_from_encoder:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.discriminator = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.Sigmoid()
        )
        self.cfg = cfg
        self.flatten = nn.Flatten(1)

    def forward(self, t_out, s_out):
        with amp.autocast(enabled=self.cfg.enable_amp):
            if self.is_feat_from_encoder:
                t_out = self.flatten(self.avg_pool(t_out))
                s_out = self.flatten(self.avg_pool(s_out))
            real_logits = self.discriminator(t_out)
            fake_logits = self.discriminator(s_out)
            loss_adv = -(torch.log(real_logits) + torch.log(1 - fake_logits)).mean()
        return loss_adv

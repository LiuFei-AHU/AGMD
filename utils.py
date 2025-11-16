import heapq
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class DistillationOrthogonalLoss(nn.Module):
    def __init__(self):
        super(DistillationOrthogonalLoss, self).__init__()

    @staticmethod
    def forward(features, features_teacher):
        flatten = nn.Flatten(1)
        features, features_teacher = flatten(features), flatten(features_teacher)
        features = F.normalize(features, p=2, dim=1)
        features_teacher = F.normalize(features_teacher, p=2, dim=1)
        dot_prod = torch.matmul(features, features.t())
        dot_prod_teacher = torch.matmul(features_teacher, features_teacher.t())
        tau = 1
        loss = abs(F.kl_div(
            dot_prod / tau,
            dot_prod_teacher / tau,
            reduction='sum',
            log_target=True
        ) * (tau * tau) / dot_prod_teacher.numel())
        return loss


def cal_metrics(confusion_matrix, average="Macro"):
    n_classes = confusion_matrix.shape[0]
    metrics_result = []
    acc, sen, spe, f1, pre = 0.0, 0.0, 0.0, 0.0, 0.0
    TP_ALL, FP_ALL, FN_ALL, TN_ALL = 0.0, 0.0, 0.0, 0.0

    for i in range(n_classes):
        ALL = np.sum(confusion_matrix)
        TP = confusion_matrix[i, i]
        FP = np.sum(confusion_matrix[:, i]) - TP
        FN = np.sum(confusion_matrix[i, :]) - TP
        TN = ALL - TP - FP - FN
        acc += TP

        TP_ALL, FP_ALL, FN_ALL, TN_ALL = TP_ALL + TP, FP_ALL + FP, FN_ALL + FN, TN_ALL + TN

        metrics_result.append([np.nan_to_num(TP / (TP + FP)),
                               np.nan_to_num(TP / (TP + FN)),
                               np.nan_to_num(TN / (TN + FP))])

    metrics_result = np.asarray(metrics_result)
    cm = np.asarray(confusion_matrix)
    acc = acc / np.sum(cm)
    if average == "Macro":
        sen = sum(metrics_result[:, 1]) / n_classes
        spe = sum(metrics_result[:, 2]) / n_classes
        pre = sum(metrics_result[:, 0]) / n_classes
    elif average == "Weighted":
        for i in range(n_classes):
            sen += metrics_result[i][1] * sum(cm[i]) / np.sum(cm)
            spe += metrics_result[i][2] * sum(cm[i]) / np.sum(cm)
            pre += metrics_result[i][0] * sum(cm[i]) / np.sum(cm)
    elif average == "Micro":
        acc = TP_ALL / (TP_ALL + FP_ALL)
        sen = TP_ALL / (TP_ALL + FN_ALL)
        spe = TN_ALL / (TN_ALL + FP_ALL)
        pre = TP_ALL / (TP_ALL + FP_ALL)

    f1 = 2 * (pre * sen) / (sen + pre)
    return acc, sen, spe, f1


def get_template(atlas_path):
    assert atlas_path is not None, "atlas_path CAN NOT BE NONE!"
    path = os.path.join(atlas_path, "raal3_{}.pt")

    templates = []
    for layer in range(4):
        template = torch.load(path.format(layer + 1)).to("cuda:0")
        templates.append(template)

    special_ROIs = []
    for temp in templates:
        template = temp.clone().to("cpu")
        special_ROI = set(torch.unique(template.flatten()).numpy())
        special_ROIs.append(special_ROI)

    private_nodes = []
    for i in range(4):
        diff = special_ROIs[i].difference(special_ROIs[i + 1]) if i < 3 else special_ROIs[i]
        private_nodes.append(diff)

    return templates, private_nodes


def get_roi_feature(feature_map_m, template_m, private_node=None):
    roi_feature = list()
    special_ROI = set(torch.unique(template_m.flatten())) if private_node is None else private_node

    for i in special_ROI:
        roi_template = (template_m == i)[np.newaxis, np.newaxis, :, :, :]
        feature = torch.sum(roi_template * feature_map_m, dim=(2, 3, 4)) / torch.sum(roi_template)
        roi_feature.append(feature)

    return roi_feature


AAL3_ADJACENT_ROIS = {
        39: [38, 42, 48, 16, 84, 22, 56, 88],
        40: [37, 39, 73, 83, 21, 55],
        41: [38, 40, 74, 84, 22, 56],
        42: [67, 68, 35, 44, 45, 46, 47, 48, 49, 51, 53],
        43: [68, 43, 45, 46, 47, 48, 50, 52, 54],
        44: [67, 68, 43, 44, 46, 49, 59],
        54: [37, 39, 41, 47, 51, 83, 53, 87, 89],
        64: [51, 85, 59, 61, 63],
        84: [65, 81, 51, 83, 53, 87, 89, 63],
    }


def add_anatomical_edges(template):
    template = [int(node) for node in template]
    anatomical_edges = []
    for roi in AAL3_ADJACENT_ROIS:
        if roi not in template:
            continue
        for neighbor in AAL3_ADJACENT_ROIS[roi]:
            if neighbor in template:
                roi_idx = template.index(roi)
                neighbor_idx = template.index(neighbor)
                anatomical_edges.append([roi_idx, neighbor_idx])

    return anatomical_edges


def build_graph_structure_batch(node_features_batch, labels, top_k=20, no_edge=False, anatomical_edges=None):
    """
    build graph for batch
    params:
        node_features_batch: nodes, shape: [batch_size, num_nodes, num_features]
        adjacency_matrix_batch: adjacent matrix，shape: [batch_size, num_nodes, num_nodes]
    return: graphs, list of Data
    """
    batch_size = node_features_batch.shape[0]
    graphs = []

    for i in range(batch_size):
        node_features = node_features_batch[i]
        adjacency_matrix = get_adjacency_matrix(node_features, k_num=top_k)
        edge_index = get_edge(adjacency_matrix)
        edge_index = torch.tensor(edge_index, dtype=torch.int64, device="cuda:0")

        if anatomical_edges:
            anatomical_edges = torch.tensor(anatomical_edges).t().contiguous()
            edge_index = torch.cat([edge_index, anatomical_edges], dim=1)

        graph = Data(x=node_features, y=None if labels is None else labels[i], edge_index=edge_index)
        graphs.append(graph)

    return graphs


def get_adjacency_matrix(roi_feature, k_num):
    roi_feature = torch.stack([roi.clone().detach().cpu() for roi in roi_feature], dim=0)
    roi_feature = F.normalize(roi_feature, p=2, dim=-1)
    feature_matrix = cosine_similarity(roi_feature)
    feature_matrix = get_binary_matrix(feature_matrix, k_num)
    return feature_matrix


def get_edge(adjacency_matrix):
    edge = list()
    roi_num = adjacency_matrix.shape[0]
    for i in range(roi_num):
        for j in range(roi_num):
            if adjacency_matrix[i, j] == 1:
                edge.append(np.array([i, j]))

    edge = np.swapaxes(np.array(edge), axis1=0, axis2=1)

    return edge


def get_binary_matrix(connection_matrix, k_num):
    roi_num = connection_matrix.shape[0]

    for i in range(roi_num):
        node_connection = connection_matrix[i, :]
        position = heapq.nlargest(k_num + 1, range(len(node_connection)), node_connection.__getitem__)
        sparse_connection = np.zeros(roi_num, dtype=np.uint8)
        for j in range(k_num + 1):
            sparse_connection[position[j]] = 1
        sparse_connection[i] = 0
        connection_matrix[i, :] = sparse_connection

    # complete connection matrix
    for i in range(roi_num):
        for j in range(roi_num):
            if connection_matrix[i, j] == 1:
                connection_matrix[j, i] = 1

    return connection_matrix


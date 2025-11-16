import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix

from AGMD.utils import cal_metrics
from AGMD.utils import AverageMeter, DistillationOrthogonalLoss
from AGMD.configs import Task, Config
from AGMD.dataset import Neuro3DDataset
from AGMD.model import TeacherModel3D, StudentModel3D, UncertaintyWeightedDistiller, GraphBuilder3D


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def cal_distill_loss(t_output, s_output, label, distiller, mode=-1, no_weight=False, discriminate_distiller=None):
    losses = {}
    weights = distiller.calc_weights(t_output)
    t_w = cal_weight(t_output, label)

    if discriminate_distiller:
        losses['mid'] = weight_func(discriminate_distiller['mid'], t_output['mid'], s_output['mid'],
                                    weights['mid'] if not no_weight else None)
    else:
        losses['mid'] = weight_func(torch.nn.MSELoss(), t_output['mid'], s_output['mid'],
                                    weights['mid'] if not no_weight else None)

    if discriminate_distiller:
        losses['deep'] = weight_func(discriminate_distiller['deep'], t_output['deep'], s_output['deep'],
                                     weights['deep'] if not no_weight else None)
    else:
        losses['deep'] = weight_func(torch.nn.MSELoss(), t_output['deep'], s_output['deep'],
                                     weights['deep'] if not no_weight else None)

    losses['graph'] = weight_func(torch.nn.MSELoss(), t_output['graph'], s_output['graph'],
                                  weights['global'] if not no_weight else None)

    if not no_weight:
        losses['global'] = (1 - F.cosine_similarity(t_output['global'], s_output['global']).mean()) * weights[
            'global'].mean()
    else:
        losses['global'] = (1 - F.cosine_similarity(t_output['global'], s_output['global']).mean())

    losses['distribution'] = weight_func(torch.nn.KLDivLoss(log_target=True, reduction="batchmean"),
                                         s_output['cls'].y, t_output['cls'].y.detach(),
                                         t_w if not no_weight else None)

    return losses


def train_student(teacher_name="", student_name="", mode=-1, task=None, best_acc_f=None, no_weight=False,
                  discriminate_distill=False):
    """
    train the student model
    :param teacher_name: the name of teacher model
    :param student_name: the name of student model
    :param mode: train mode. -1: complete distillation, 0: no distillation
    :param task: task name
    :param best_acc_f: threshold to save the model parameters
    :param no_weight: whether to use gating weight
    :param discriminate_distill: whether to use discriminator
    :return: None
    """

    if task is None:
        task = Task.task1
    cfg = Config()
    cfg.num_classes = task["num_class"]
    cfg.task_no = task["Task_no"]
    cfg.task = task
    cfg.mode = mode

    teacher = TeacherModel3D(cfg).to(cfg.device)
    student = StudentModel3D(cfg).to(cfg.device)
    distiller = UncertaintyWeightedDistiller()
    GraphBuilder3D.init(cfg)

    load_model(teacher, cfg, teacher_name)
    load_model(student, cfg, student_name)

    train_list = torch.nn.ModuleList()
    train_list.append(student)

    discriminate_distiller = None
    if discriminate_distill:
        d_distiller = {}
        if mode == -1:
            d_distiller["mid"] = DistillationOrthogonalLoss().to(
                cfg.device)  # DiscriminatorDistiller(encoder_fet=True, in_dim=256, cfg=cfg).to(cfg.device)
            d_distiller["deep"] = DistillationOrthogonalLoss().to(
                cfg.device)  # DiscriminatorDistiller(encoder_fet=True, in_dim=512, cfg=cfg).to(cfg.device)
            d_distiller["global"] = DistillationOrthogonalLoss().to(
                cfg.device)  # DiscriminatorDistiller(encoder_fet=True, in_dim=512, cfg=cfg).to(cfg.device)
        discriminate_distiller = nn.ModuleDict(d_distiller)
        train_list.append(discriminate_distiller)

    loader, test_loder = load_data(cfg)
    optimizer = torch.optim.AdamW(train_list.parameters(), lr=cfg.lr, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    scheduler = None
    scaler = amp.GradScaler(enabled=cfg.enable_amp)

    loss_meter = AverageMeter('Loss', ':.4e')
    acc_meter = AverageMeter('Acc', ':6.2f')

    if best_acc_f:
        best_acc = best_acc_f
        min_loss = 100.0
    else:
        if student_name:
            itm = student_name.split("_")
            best_acc, min_loss = float(itm[-2]), float(itm[-1])

    # Start to train the model
    for epoch in range(cfg.num_epochs):
        total_loss = 0.0
        teacher.eval()
        student.train()

        for mri, pet, label in loader:
            optimizer.zero_grad()
            with amp.autocast(enabled=cfg.enable_amp):
                mri = mri.to(cfg.device)
                pet = pet.to(cfg.device)
                label = label.to(cfg.device)
                with torch.no_grad():
                    t_output = teacher(mri, pet, label)
                s_output = student(mri, label)

            if mode == 0:
                total_batch_loss = s_output["cls"].loss
            else:
                losses = cal_distill_loss(t_output, s_output, label, distiller, mode, no_weight,
                                          discriminate_distiller)
                total_batch_loss = sum(losses.values()) + s_output["cls"].loss

            acc = accuracy(s_output["cls"].y, label)
            acc_meter.update(acc, len(label))
            loss_meter.update(total_batch_loss.item(), len(label))
            scale_backward(scaler, total_batch_loss, optimizer, student)
            total_loss += total_batch_loss.item()

        # Test
        student.eval()
        acc_meter.reset()
        loss_meter.reset()

        with torch.no_grad():
            for mri, _, label in test_loder:
                with amp.autocast(enabled=cfg.enable_amp):
                    mri = mri.to(cfg.device)
                    label = label.to(cfg.device)
                    s_output = student(mri, label)
                acc = accuracy(s_output["cls"].y, label)
                acc_meter.update(acc, len(label))
                loss_meter.update(s_output["cls"].loss.item(), len(label))

        val_acc, val_loss = acc_meter.avg, loss_meter.avg
        print(
            f"Epoch [{epoch + 1}/{cfg.num_epochs}] "
            f"[Train] Loss: {total_loss / len(loader):.4f} Acc: {acc_meter.avg:.4f} "
            f"[Test] Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # save
        if val_acc > best_acc or (val_acc == best_acc and val_loss < min_loss):
            min_loss = val_loss if val_loss < min_loss else min_loss
            best_acc = val_acc
            sub_fix = "_NW" if no_weight else ""
            if mode == -1:
                best_save_path = cfg.checkpoint_path + f'/[S]Task-{cfg.task_no}{sub_fix}_epoch_{epoch}_{val_acc:.4f}_{val_loss:.4f}'
            else:
                best_save_path = cfg.checkpoint_path + f'/[S]Task-{cfg.task_no}_mode_{str(mode)}{sub_fix}_epoch_{epoch}_{val_acc:.4f}_{val_loss:.4f}'
            torch.save(student.state_dict(), best_save_path + '.pdparams')

        if scheduler:
            scheduler.step()


def train_teacher(model_name="", task=None, best_acc_f=None, cmt=True, anatomical_graph=True):
    """
    train the teacher model
    :param model_name: the name of teacher model
    :param task: task name
    :param best_acc_f: threshold to save the model parameters
    :param cmt: whether to use cross-modal attention (CMT)
    :param anatomical_graph: whether to use anatomical graph
    :return: None
    """
    if task is None:
        task = Task.task1
    cfg = Config()
    cfg.num_classes = task["num_class"]
    cfg.task_no = task["Task_no"]
    cfg.task = task
    cfg.model_name = model_name
    cfg.CMT = cmt
    cfg.anatomical_graph = anatomical_graph

    subfix = []
    if not cmt:
        subfix.append("ncmt")
    if not anatomical_graph:
        subfix.append("nana")

    subfix = "_" + "_".join(subfix)

    os.makedirs(cfg.checkpoint_path, exist_ok=True)

    teacher = TeacherModel3D(cfg).to(cfg.device)
    load_model(teacher, cfg, model_name)

    optimizer = torch.optim.Adam(teacher.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
    scaler = amp.GradScaler(enabled=cfg.enable_amp)
    GraphBuilder3D.init(cfg)

    loader, test_loder = load_data(cfg)

    loss_meter = AverageMeter('Loss', ':.4e')
    acc_meter = AverageMeter('Acc', ':6.2f')

    if best_acc_f:
        best_acc = best_acc_f
        min_loss = 100.0
    else:
        if model_name:
            itm = model_name.split("_")
            best_acc, min_loss = float(itm[-2]), float(itm[-1])

    # Start to train the model
    for epoch in range(cfg.num_epochs):
        total_loss = 0.0
        teacher.train()
        for mri, pet, label in loader:
            optimizer.zero_grad()
            with amp.autocast(enabled=cfg.enable_amp):
                mri = mri.to(cfg.device)
                pet = pet.to(cfg.device)
                label = label.to(cfg.device)
                t_output = teacher(mri, pet, label)
            total_batch_loss = t_output["cls"].loss
            acc = accuracy(t_output["cls"].y, label)

            acc_meter.update(acc, len(label))
            loss_meter.update(total_batch_loss.item(), len(label))
            scale_backward(scaler, total_batch_loss, optimizer, teacher)
            total_loss += total_batch_loss.item()

        # Test
        teacher.eval()
        acc_meter.reset()
        loss_meter.reset()

        with torch.no_grad():
            for mri, pet, label in test_loder:
                with amp.autocast(enabled=cfg.enable_amp):
                    mri = mri.to(cfg.device)
                    pet = pet.to(cfg.device)
                    label = label.to(cfg.device)
                    t_output = teacher(mri, pet, label)
                acc = accuracy(t_output["cls"].y, label)
                acc_meter.update(acc, len(label))
                loss_meter.update(t_output["cls"].loss.item(), len(label))

        val_acc, val_loss = acc_meter.avg, loss_meter.avg
        print(
            f"Epoch [{epoch + 1}/{cfg.num_epochs}] "
            f"[Train] Loss: {total_loss / len(loader):.4f} Acc: {acc_meter.avg:.4f} "
            f"[Test] Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # save
        if val_acc > best_acc or (val_acc == best_acc and val_loss < min_loss):
            min_loss = val_loss if val_loss < min_loss else min_loss
            best_acc = val_acc
            best_save_path = cfg.checkpoint_path + f'/[T]Task-{cfg.task_no}{subfix}_epoch_{epoch}_{val_acc:.4f}_{val_loss:.4f}'
            torch.save(teacher.state_dict(), best_save_path + '.pdparams')

        if scheduler:
            scheduler.step()


def test_teacher(model_name, task, cmt=True, anatomical_graph=True):
    cfg = Config()
    cfg.task = task
    cfg.num_classes = task["num_class"]
    cfg.training = True
    cfg.CMT = cmt
    cfg.anatomical_graph = anatomical_graph

    _, test_loader = load_data(cfg)
    model = TeacherModel3D(cfg).to(cfg.device)
    model.eval()
    GraphBuilder3D.init(cfg)
    load_model(model, cfg, model_name)

    preds, target = [], []
    auc_preds = []
    softmax = torch.nn.Softmax()

    with torch.no_grad():
        for mri, pet, label in test_loader:
            with amp.autocast(enabled=cfg.enable_amp):
                mri = mri.to(cfg.device)
                pet = pet.to(cfg.device)
                label = label.to(cfg.device)
                s_output = model(mri, pet, label)
            pred = softmax(s_output['cls'].p)
            preds.extend(pred.cpu().detach().max(1)[1])
            auc_preds.extend(pred)
            target.extend(label.cpu().detach())

    auc_preds = [auc_preds[i].cpu().detach().numpy() for i in range(len(auc_preds))]
    cm = confusion_matrix(target, preds)
    acc, sen, spe, f1 = cal_metrics(cm, "Weighted")
    auc = roc_auc_score(target, auc_preds, multi_class='ovr')

    print("acc:{:.4f} auc:{:.4f} sen:{:.4f} spe:{:.4f} f1:{:.4f}".format(acc, auc, sen, spe, f1))
    return preds, target, (acc, auc, sen, spe, f1)


def test_student(model_names, task=None):
    """
    test the student model
    :param model_names: list of the name of trained models
    :param task: task name
    :return:
    """
    cfg = Config()
    cfg.task = task
    cfg.num_classes = task["num_class"]
    cfg.training = False
    _, test_loader = load_data(cfg)

    if isinstance(model_names, str):
        model_names = [model_names]

    for m_name in model_names:
        print("-" * 10 + m_name + "-" * 10)
        student = StudentModel3D(cfg).to(cfg.device)
        GraphBuilder3D.init(cfg)
        load_model(student, cfg, m_name)

        test(student, test_loader, cfg)


def test(model, data_loader, cfg):
    preds, target = [], []
    auc_preds = []
    softmax = torch.nn.Softmax()
    model.eval()
    with torch.no_grad():
        for mri, _, label in data_loader:
            with amp.autocast(enabled=cfg.enable_amp):
                mri = mri.to(cfg.device)
                label = label.to(cfg.device)
                s_output = model(mri, label)
            pred = s_output['cls']
            pred = softmax(pred)
            preds.extend(pred.cpu().detach().max(1)[1])
            auc_preds.extend(pred)
            target.extend(label.cpu().detach())

    auc_preds = [auc_preds[i].cpu().detach().numpy() for i in range(len(auc_preds))]
    cm = confusion_matrix(target, preds)
    acc, sen, spe, f1 = cal_metrics(cm, "Weighted")
    auc = roc_auc_score(target, auc_preds, multi_class='ovr')
    print("acc:{:.4f} auc:{:.4f} sen:{:.4f} spe:{:.4f} f1:{:.4f}".format(acc, auc, sen, spe, f1))
    return preds, target, (acc, auc, sen, spe, f1)


def cal_weight(out_t, label):
    weight = abs(out_t['cls'].p.max(1)[0] - label)
    weight[weight > 1] = 1
    weight = 1 - weight
    return weight


def weight_func(func, x, y, weight):
    if weight is None:
        return func(x, y).mean()
    return (func(x, y) * weight).mean()


def uncertainty_evaluate(t_output):
    weights = {}
    probabilities = t_output["cls"].p
    for level in ['mid', 'deep', 'global']:
        score = uncertainty_score(probabilities)
        weights[level] = score

    return weights


def uncertainty_score(probabilities):
    entropy = calculate_entropy(probabilities)
    return 1 - entropy


def calculate_entropy(probabilities, n_class=4.0):
    epsilon = 1e-10
    return -torch.sum(probabilities * torch.log(probabilities + epsilon), dim=1) / np.log(n_class)


def cosine_similarity(feature_teacher, feature_student):
    """
    calculate the features' cosine similarity
    :param feature_teacher: teacher model's features
    :param feature_student: student model's features
    :return: shape like (n_samples,)
    """
    b, c = feature_teacher.size(0), feature_teacher.size(1)
    feature_teacher_normalized = F.normalize(feature_teacher.view(b, c, -1), p=2, dim=2)
    feature_student_normalized = F.normalize(feature_student.view(b, c, -1), p=2, dim=2)

    similarity = torch.sum(feature_teacher_normalized * feature_student_normalized, dim=2)
    return similarity


def scale_backward(scaler, loss, optimizer, model):
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


def load_data(cfg):
    data = np.load(cfg.dataset)
    mris, labels, pets = data['mris'], data['labels'], data['pets']
    print("AD", len(labels[labels == 3]), "PMCI", len(labels[labels == 2]), "SMCI", len(labels[labels == 1]), "CN",
          len(labels[labels == 0]))

    dataset = Neuro3DDataset(mris, labels, pets)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
    return loader, loader


def load_model(model, cfg, model_name):
    params_file_path = os.path.join(cfg.checkpoint_path, model_name + ".pdparams")
    if os.path.exists(params_file_path) and os.path.isfile(params_file_path):
        model.load_state_dict(torch.load(params_file_path, map_location=cfg.device), strict=True)
        print('Loaded!')





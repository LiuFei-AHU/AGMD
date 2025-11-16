"""
For the paper in MICCAI 2025:
"Anatomical Graph-based Multilevel Distillation for Robust Alzheimer's Disease Diagnosis with Missing Modalities".
URL: https://papers.miccai.org/miccai-2025/paper/3438_paper.pdf
"""
import argparse
from AGMD.configs import Task
from AGMD.train_test import train_teacher, train_student, test_teacher, test_student


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='task name', default="task1")
    parser.add_argument('--run_mode', type=str, help='running mode: train, test', default="train")
    parser.add_argument('--mode', type=int, help='', default=-1)
    parser.add_argument('--run_teacher', type=str2bool, help='run teacher or student', default="y")
    parser.add_argument('--teacher_name', type=str, help='name of trained teacher model', default="")
    parser.add_argument('--student_name', type=str, help='name of trained student model', default="")
    parser.add_argument('--no_weight', type=str2bool, help='', default="n")
    parser.add_argument('--cmt', type=str2bool, help='', default="y")
    parser.add_argument('--anatomical_graph', type=str2bool, help='', default="y")

    args = parser.parse_args()
    if args.task == "task1":
        args.task = Task.task1
    else:
        args.task = Task.task2

    return args


def str2bool(v):
    """
    true/false, yes/no, y/n, 1/0
    """
    if isinstance(v, bool):
        return v
    v_lower = v.strip().lower()
    if v_lower in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v_lower in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"invalid value: '{v}'")


def train(arg):
    if arg.run_teacher:
        train_teacher(model_name=arg.teacher_name, task=arg.task, best_acc_f=0.5, cmt=arg.cmt,
                      anatomical_graph=arg.anatomical_graph)
    else:
        train_student(teacher_name=arg.teacher_name, student_name=arg.student_name, mode=arg.mode,
                      task=arg.task, best_acc_f=0.5, no_weight=arg.no_weight,
                      discriminate_distill=arg.discriminate_distill)


def test(arg):
    if arg.run_teacher:
        test_teacher(model_name=arg.teacher_name, task=arg.task, cmt=arg.cmt, anatomical_graph=arg.anatomical_graph)
    else:
        test_student(model_names=[arg.student_name], task=arg.task)


if __name__ == "__main__":
    args = parse_args()

    if args.run_mode == "train":
        train(args)
    if args.run_mode == "test":
        test(args)

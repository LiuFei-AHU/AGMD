import torch


class Task:
    task1 = {"Task_no": 1, "name": "AD-NC", "num_class": 2}
    task2 = {"Task_no": 2, "name": "AD-pMCI-sMCI-NC", "num_class": 4}


class Config:
    batch_size = 8  # u can use your CNN encoder, a small encoder is ok. if so, u can train more data in one batch!
    num_epochs = 150
    lr = 1e-5
    num_classes = 4  # here is a default value, it is determined by the specific task
    training = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enable_amp = True if torch.cuda.is_available() else False
    input_shape = (113, 137, 113)  # U can set to your image shape, do not forget to adjust the atlas!
    # here the data is saved as a npy zip format. for the dataset format details, pls see 'readme'.
    # u can use your own dataset, u should implement the load_data() in train_test.py
    dataset = './data/demo.npz'  # TODO set your dataset path
    checkpoint_path = './out'
    atlas_path = "./data/atlas"

    anatomical_graph = True
    CMT = True

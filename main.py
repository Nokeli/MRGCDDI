import torch
import numpy as np

from parms_setting import settings
from data_preprocess import load_data, generate_data
from instantiation import Create_model
from train import train_model
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_random_seed(2023, deterministic=False)

# parameters setting
args = settings()

args.cuda = not args.no_cuda and torch.cuda.is_available()


data_o, data_s, data_a, train_loader, val_loader, test_loader = load_data(args)
train_data,vaild_data,test_data = generate_data(args)
args.test_data = test_data
args.vaild_data = vaild_data
args.train_data = train_data

# train and test model
model, optimizer = Create_model(args)
print(model)
train_model(model, optimizer, data_o, data_s, data_a, train_loader, val_loader, test_loader, args,train_data)


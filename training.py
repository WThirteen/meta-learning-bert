import time
import logging
import os
import torch
import random
import numpy as np
import Config as cg

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

def random_seed(value):
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    np.random.seed(value)
    random.seed(value)

def create_batch_of_tasks(taskset, is_shuffle = True, batch_size = 4):
    idxs = list(range(0,len(taskset)))
    if is_shuffle:
        random.shuffle(idxs)
    for i in range(0,len(idxs), batch_size):
        yield [taskset[idxs[i]] for i in range(i, min(i + batch_size,len(taskset)))]


class TrainingArgs:
    def __init__(self):
        self.num_labels = cg.num_labels
        self.meta_epoch = cg.meta_epoch
        self.k_spt = cg.k_support
        self.k_qry = cg.k_query
        self.outer_batch_size = cg.outer_batch_size
        self.inner_batch_size = cg.inner_batch_size
        self.outer_update_lr = cg.outer_update_lr
        self.inner_update_lr = cg.inner_update_lr
        self.inner_update_step = cg.inner_update_step
        self.inner_update_step_eval = cg.inner_update_step_eval
        self.bert_model = cg.model_path
        self.num_task_train = cg.train_num_task
        self.num_task_test = cg.test_num_task


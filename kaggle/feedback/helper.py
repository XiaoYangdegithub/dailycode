import math
import time
from cfg import CFG
import os
import random
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, f1_score, confusion_matrix

class AverageMeter(object):  #设置的指标，算loss的时候用的
    """Computes and stores the average and current value"""

    def __init__(self):
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


def asMinutes(s):  #设置的时间
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


"""## logger"""


def get_logger(filename=CFG.OUTPUT_DIR + 'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def seed_everything(seed=42):  #设置随机种子
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def MCRMSE(y_trues, y_preds): #设置评价指标
    """
    Credits to Y. Nakama for this function:
    https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train?scriptVersionId=104639699&cellId=10
    """
    y_trues = np.asarray(y_trues)
    y_preds = np.asarray(y_preds)
    print(y_trues)
    print(y_preds)
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        scores.append(rmse)
    mcrmse_score = np.mean(scores)
    return mcrmse_score

def compute_metrics(p):
    predictions, labels = p
    scores = []
    idxes = labels.shape[1]
    for i in range(idxes):
        y_true = labels[:, i]
        y_pred = predictions[:, i]
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        scores.append(rmse)
    mcrmse_score = np.mean(scores)
    return {'mcrmse':  mcrmse_score }
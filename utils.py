import numpy as np
import random
import torch
import torch.nn.functional as F
from datasets.data_utils import split_ssl_data
import json
from datasets.dataset import BasicDataset
from datasets.ssl_dataset import get_transform
mean, std = {}, {}
mean['cifar10'] = [0.4914, 0.4822, 0.4465]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]

std['cifar10'] = [0.2023, 0.1994, 0.2010]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]
def set_global_seeds(i):
    random.seed(i)
    torch.manual_seed(i)
    np.random.seed(i)


def set_device():
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")
    print(f'Current device is {_device}', flush=True)
    return _device


# Adjust learning rate and for SGD Optimizer
def adjust_learning_rate(optimizer, epoch,alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def hier_score(label_map,log1,log2):
    '''
        logics: tensor [batch_size,fine_label_size]
        label_map: low_layer_label: high_layer_label

        return score: list batch_size
    '''
    scores=[]
    logics=F.softmax(log1, dim=1)
    logics_2=F.softmax(log2, dim=1)
    for i,logic in enumerate(logics):
        true_label=torch.argmax(logic)
        scores.append(float(logic[true_label]*logics_2[i][label_map[true_label]]))
    return scores

import os
import time
from torch.utils.tensorboard import SummaryWriter
import logging
import yaml


def over_write_args_from_file(args, yml):
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])


def setattr_cls_from_kwargs(cls, kwargs):
    # if default values are in the cls,
    # overlap the value by kwargs
    for key in kwargs.keys():
        if hasattr(cls, key):
            print(f"{key} in {cls} is overlapped by kwargs: {getattr(cls, key)} -> {kwargs[key]}")
        setattr(cls, key, kwargs[key])


def test_setattr_cls_from_kwargs():
    class _test_cls:
        def __init__(self):
            self.a = 1
            self.b = 'hello'

    test_cls = _test_cls()
    config = {'a': 3, 'b': 'change_hello', 'c': 5}
    setattr_cls_from_kwargs(test_cls, config)
    for key in config.keys():
        print(f"{key}:\t {getattr(test_cls, key)}")


def net_builder(net_name, from_name: bool, net_conf=None, is_remix=False):
    """
    return **class** of backbone network (not instance).
    Args
        net_name: 'WideResNet' or network names in torchvision.models
        from_name: If True, net_buidler takes models in torch.vision models. Then, net_conf is ignored.
        net_conf: When from_name is False, net_conf is the configuration of backbone network (now, only WRN is supported).
    """
    if from_name:
        import torchvision.models as models
        model_name_list = sorted(name for name in models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(models.__dict__[name]))

        if net_name not in model_name_list:
            assert Exception(f"[!] Networks\' Name is wrong, check net config, \
                               expected: {model_name_list}  \
                               received: {net_name}")
        else:
            return models.__dict__[net_name]

    else:
        if net_name == 'WideResNet':
            import models.nets.wrn as net
            builder = getattr(net, 'build_WideResNet')()
        elif net_name == 'WideResNetVar':
            import models.nets.wrn_var as net
            builder = getattr(net, 'build_WideResNetVar')()
        elif net_name == 'ResNet50':
            import models.nets.resnet50 as net
            builder = getattr(net, 'build_ResNet50')(is_remix)
        elif net_name == 'ResNet18':
            import models.nets.resnet as net
            builder = getattr(net, 'build_ResNet18')()
        else:
            assert Exception("Not Implemented Error")

        if net_name != 'ResNet50':
            setattr_cls_from_kwargs(builder, net_conf)
        return builder.build


def test_net_builder(net_name, from_name, net_conf=None):
    builder = net_builder(net_name, from_name, net_conf)
    print(f"net_name: {net_name}, from_name: {from_name}, net_conf: {net_conf}")
    print(builder)


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def get_high_confidence_index(args,loader, model):
    
    loss_s = torch.Tensor([]).cuda(args.gpu)
    label_s = torch.Tensor([]).cuda(args.gpu)
    index_pres = torch.Tensor([]).cuda(args.gpu)
    
    with torch.no_grad():
        model.eval()    # Change model to 'eval' mode.
        
        for images, labels, indexs in loader:
            images = images.cuda(args.gpu)
            labels = labels.cuda(args.gpu)
            indexs = indexs.cuda(args.gpu)
            logits = model(images)
            
            outputs = F.cross_entropy(logits,labels,reduce=False)
            # print("outputs:", outputs)
            
            loss_s = torch.cat((loss_s, outputs), 0)
            label_s = torch.cat((label_s, labels), 0)
            index_pres = torch.cat((index_pres, indexs), 0)
            
    
    if args.dataset == 'cifar10':
        num_classes = 10
    else:
        num_classes = 100
    ans = torch.Tensor([]).cuda(args.gpu)
    for i in range(num_classes):
        index = torch.where(label_s == i)[0]
        loss_first = torch.index_select(loss_s, 0, index)
        index_first = torch.index_select(index_pres, 0, index)
        # print('probs_first:', loss_first)
        # print(loss_first.shape,args.number_sample)
        values, indices = loss_first.topk(args.number_sample, dim=0, largest=False, sorted=True)
        ansnow = torch.index_select(index_first, 0, indices)
        
        ans = torch.cat((ans, ansnow), 0)
    ans = ans.to(torch.long)
    # print(torch.index_select(loss_s, 0, ans))
    ans = ans.cpu().numpy()

        
    
    return ans

def train(args, epoch, train_loader, model, optimizer):
    train_total=0
    train_correct=0
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for i, (images, labels, indexes) in enumerate(train_loader):

        batch_size = indexes.shape[0]
       
        images =images.cuda(args.gpu)
        labels =labels.cuda(args.gpu)
       
        # Forward + Backward + Optimize
        logits = model(images)

        prec, _ = accuracy(logits, labels, topk=(1, 5))
        train_total+=1
        train_correct+=prec
        loss = criterion(logits, labels)
        
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    train_acc=float(train_correct)/float(train_total)
    return train_acc

def get_ssl_dset(args, num_labels, index=None, include_lb_to_ulb=True, 
            strong_transform=None,onehot=False, data=None, targets=None):
    """
    get_ssl_dset split training samples into labeled and unlabeled samples.
    The labeled data is balanced samples over classes.
    """

    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, data, targets,
        num_labels, args.num_classes, index, include_lb_to_ulb)
    
    # output the distribution of labeled data for remixmatch
    count = [0 for _ in range(args.num_classes)]
    for c in lb_targets:
        count[c] += 1
    dist = np.array(count, dtype=float)
    dist = dist / dist.sum()
    dist = dist.tolist()
    out = {"distribution": dist}
    output_file = r"./data_statistics/"
    output_path = output_file + str(args.name) + '_' + str(num_labels) + '.json'
    if not os.path.exists(output_file):
        os.makedirs(output_file, exist_ok=True)
    with open(output_path, 'w') as w:
        json.dump(out, w)
    # print(Counter(ulb_targets.tolist()))
    transform = get_transform(mean[args.name], std[args.name], 32, train)
    lb_dset = BasicDataset(args.alg, lb_data, lb_targets, args.num_classes,
                            transform, False, None, onehot)
    
    # assert 0
    ulb_dset = BasicDataset(args.alg, ulb_data, ulb_targets, args.num_classes,
                            transform, True, strong_transform, onehot)
    # print(lb_data.shape)
    # print(ulb_data.shape)
    return lb_dset, ulb_dset


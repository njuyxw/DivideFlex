# -*- coding:utf-8 -*-

import os

import torch.nn as nn

import torch
import random
import numpy as np

from utils import net_builder
from models.resnet.resnet import PreResNet18
from data.datasets import input_dataset

def set_global_seeds(i):
    random.seed(i)
    torch.manual_seed(i)
    np.random.seed(i)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, default='./saved_models/divideflex_worst/seed_2_20220606-094818/model_last.pth')
    parser.add_argument('--use_train_model', action='store_true')
    parser.add_argument('--seed', default=0, type=int)

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--net_from_name', type=bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)

    '''
    Data Configurations
    '''
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num_classes', type=int, default=10)

    parser.add_argument('--val_ratio', type = float, default = 0)
    parser.add_argument('--noise_type', type = str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='worst')
    parser.add_argument('--noise_path', type = str, help='path of CIFAR-10_human.pt', default=None)

    args = parser.parse_args()
    noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
    args.noise_type = noise_type_map[args.noise_type]
    set_global_seeds(args.seed)
    # load dataset
    if args.noise_path is None:
        if args.dataset == 'cifar10':
            args.noise_path = './data/CIFAR-10_human.pt'
        elif args.dataset == 'cifar100':
            args.noise_path = './data/CIFAR-100_human.pt'
        else: 
            raise NameError(f'Undefined dataset {args.dataset}')
    checkpoint_path = os.path.join(args.load_path)
    checkpoint = torch.load(checkpoint_path)
     
    load_model = checkpoint['ema_model'] 
    print(load_model.keys())
    # _net_builder = net_builder(args.net, 
    #                            args.net_from_name,
    #                            {'depth': args.depth, 
    #                             'widen_factor': args.widen_factor,
    #                             'leaky_slope': args.leaky_slope,
    #                             'dropRate': args.dropout,
    #                             'use_embed': False})
    # _net_builder = PreResNet18()
    
    net = PreResNet18(num_classes=args.num_classes)
    net = nn.DataParallel(net)
    net.load_state_dict(load_model)
    if torch.cuda.is_available():
        net.cuda()
    train_dataset, val_dataset, test_dataset, num_classes, num_training_samples = input_dataset(args.dataset,args.noise_type, args.noise_path, is_human = True, val_ratio = args.val_ratio)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                  batch_size = 128,
                                  num_workers=8,
                                  shuffle=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                  batch_size = 64,
                                  num_workers=8,
                                  shuffle=True)

 
    acc = 0.0
    net.eval()
    with torch.no_grad():
        for image, target,_ in test_loader :
            image = image.type(torch.FloatTensor).cuda()
            logit = net(image)
            
            acc += logit.cpu().max(1)[1].eq(target).sum().numpy()
    


    print(f'Best test acc is {acc/len(test_dataset)*100}%')


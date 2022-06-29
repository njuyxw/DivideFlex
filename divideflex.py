import torch
import argparse
import os
import copy
import random
import numpy as np
import torch.backends.cudnn as cudnn
import logging
import time
import torch.nn.functional as F
from data.datasets import input_dataset
from models.resnet.resnet import ResNet34, PreResNet18
from torchvision import transforms
from sklearn.mixture import GaussianMixture


from utils import train, get_high_confidence_index, net_builder, get_logger, count_parameters, over_write_args_from_file, get_ssl_dset

from train_utils import TBLog, get_optimizer, get_cosine_schedule_with_warmup


from datasets.ssl_dataset import get_transform
from flexmatch.flexmatch import FlexMatch

from datasets.data_utils import get_data_loader

# def divide(args, logger, save_path, train_dataset, num_classes):    
#     # load model
#     print('building model...')
#     model = ResNet34(num_classes).cuda(args.gpu)
#     print('building model done')

#     optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
#         momentum=0.9, weight_decay=5e-4)
#     # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

#     train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
#         batch_size = 128,
#         num_workers=8,
#         shuffle=False
#     )
#     for epoch in range(args.pre_epochs):
#         train_acc = train(args,epoch, train_loader, model, optimizer)
#         logger.info(f'pre epoch {epoch}, train acc with noise {train_acc}')
#     high_confidence_index = get_high_confidence_index(args,loader=train_loader, model=model)
    
#     idnum= 0
#     for idx in high_confidence_index:
#         if train_dataset.train_labels[idx]==train_dataset.train_noisy_labels[idx]:
#             idnum+=1 
#     logger.info('semi-model select acc: %f'%(100*idnum/len(high_confidence_index)))
    
#     np.save(save_path + "/high_confidence_index.npy", high_confidence_index)
#     return high_confidence_index

def divide(args, logger, save_path, train_dataset, num_classes):
    # load model
    print('building model...')
    model1 = PreResNet18(num_classes).cuda(args.gpu)
    model2 = PreResNet18(num_classes).cuda(args.gpu)
    model3 = PreResNet18(num_classes).cuda(args.gpu)
    print('building model done')

    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4)
    optimizer3 = torch.optim.SGD(model3.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size=128,
        num_workers=8,
        shuffle=True
    )
    for epoch in range(args.pre_epochs):
        train_acc1 = train(args,epoch, train_loader, model1, optimizer1)
        train_acc2 = train(args,epoch, train_loader, model2, optimizer2)
        train_acc3 = train(args,epoch, train_loader, model3, optimizer3)
        logger.info(f'pre epoch {epoch}, train acc1 with noise {train_acc1}')
        logger.info(f'pre epoch {epoch}, train acc2 with noise {train_acc2}')
        logger.info(f'pre epoch {epoch}, train acc3 with noise {train_acc3}')

    test_cifar10_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # get data
    # eval_dataset, _, _, _, _ = input_dataset(args.dataset,args.noise_type, args.noise_path, 
    #     is_human = True, val_ratio = args.val_ratio)
    # eval_dataset.transform = test_cifar10_transform
    eval_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size=128,
        num_workers=8,
        shuffle=False
    )

    loss1 = eval_train(model1, eval_loader)
    loss2 = eval_train(model2, eval_loader)
    loss3 = eval_train(model3, eval_loader)
    losses = loss1 + loss2 + loss3

    pred_clean_num = fit_gmm(losses, logger)
    select_sample_num_pre_class = round(args.lambda_r * pred_clean_num)

    high_confidence_index = loss_divide(losses, args, train_dataset, logger, select_sample_num_pre_class)

    # high_confidence_index = get_high_confidence_index(args,loader=train_loader, model=model)
    
    ### 
    idnum= 0
    for idx in high_confidence_index:
        if train_dataset.train_labels[idx]==train_dataset.train_noisy_labels[idx]:
            idnum+=1 
    logger.info('semi-model select acc: %f'%(100*idnum/len(high_confidence_index)))
    
    np.save(save_path + "/high_confidence_index.npy", high_confidence_index)
    return high_confidence_index

def fit_gmm(losses, logger):
    input_loss = losses.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]
    pred_clean_num = (prob > 0.5).sum()
    logger.info(f"gmm predict clean sample num: {pred_clean_num}")   
    return pred_clean_num


def eval_train(model, eval_loader):    
    model.eval()
    losses = torch.zeros(50000)    
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets, reduction='none')  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]         
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    return losses

def loss_divide(losses, args, train_set, logger, select_sample_num_pre_class):
    noise_label = np.array(train_set.train_noisy_labels)
    clean_label = np.array(train_set.train_labels)
    is_clean = (noise_label == clean_label)
    losses = np.array(losses)
    class_select_num = select_sample_num_pre_class
    select_index = []
    precision = []
    for i in range(args.num_classes):
        class_index = np.where(noise_label == i)[0]
        class_losses = losses[class_index]
        sorted_index = np.argsort(class_losses)[0:class_select_num]
        select_index.append(class_index[sorted_index])
        precision.append(is_clean[select_index[i]].sum()/class_select_num)

    select_index = np.array(select_index).reshape(class_select_num*args.num_classes , )
    # select_true = (np.zeros(50000) != 0)
    # select_true[select_index] = True  
    logger.info(f"clean precision: {precision}, mean precision: {np.array(precision).mean()}")
    logger.info(f"select sample number: {len(select_index)}")
    return select_index
    

def main_worker(args):

    # random seed has to be set for the syncronization of labeled data sampling in each process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    cudnn.deterministic = True
    
    # gpu setting
    torch.cuda.set_device(args.gpu)
    
    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    nowtime = time.strftime('%Y%m%d-%H%M%S',time.localtime(time.time()))
    save_path = save_path + '/seed_%d_' % args.seed + nowtime
    args.save_path = save_path
    # print('save_path ',save_path)
    
    tb_log = TBLog(args.save_path, 'tensorboard', use_tensorboard=args.use_tensorboard)
    logger_level = "INFO"

    logger = get_logger(args.save_name, args.save_path, logger_level)
    logger.warning(f"USE GPU: {args.gpu} for training")

    # SET flexmatch: class flexmatch in models.flexmatch
    # _net_builder = net_builder(args.net,
    #     args.net_from_name,
    #     {'first_stride': 2 if 'stl' in args.dataset else 1,
    #     'depth': args.depth,
    #     'widen_factor': args.widen_factor,
    #     'leaky_slope': args.leaky_slope,
    #     'bn_momentum': 0.0001,
    #     'dropRate': args.dropout,
    #     'use_embed': False,
    #     'is_remix': False},
    # )



    model = FlexMatch(PreResNet18,
        args.num_classes,
        args.ema_m,
        args.T,
        args.p_cutoff,
        args.ulb_loss_ratio,
        args.hard_label,
        num_eval_iter=args.num_eval_iter,
        tb_log=tb_log,
        logger=logger
    )

    logger.info(f'Number of Trainable Params: {count_parameters(model.model)}')

    # SET Optimizer & LR Scheduler
    ## construct SGD and cosine lr scheduler
    optimizer = get_optimizer(model.model, args.optim, args.lr, args.momentum, args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
        args.num_train_iter,
        num_warmup_steps=args.num_train_iter * 0
    )
    ## set SGD and cosine lr on flexmatch
    model.set_optimizer(optimizer, scheduler)

    # model and ema_model setting
    model.model = model.model.cuda(args.gpu)
    model.model = torch.nn.DataParallel(model.model).cuda()
    model.ema_model = copy.deepcopy(model.model)

    logger.info(f"model_arch: {model}")
    logger.info(f"Arguments: {args}")

    # get data
    train_dataset, _, test_dataset, num_classes, _ = input_dataset(args.dataset,args.noise_type, args.noise_path, 
        is_human = True, val_ratio = args.val_ratio)

    high_confidence_index = divide(args, logger, args.save_path, train_dataset, num_classes)

    
    # get the training dataset and test(eval) dataset
    lb_dset, ulb_dset = get_ssl_dset(args, args.num_labels, index=high_confidence_index, data=train_dataset.train_data, targets=train_dataset.train_noisy_labels)
    eval_dset = test_dataset

    
    loader_dict = {}
    dset_dict = {'train_lb': lb_dset, 'train_ulb': ulb_dset, 'eval': eval_dset}

    loader_dict['train_lb'] = get_data_loader(dset_dict['train_lb'],
        args.batch_size,
        data_sampler=args.train_sampler,
        num_iters=args.num_train_iter,
        num_workers=args.num_workers
    )

    loader_dict['train_ulb'] = get_data_loader(dset_dict['train_ulb'],
        args.batch_size * args.uratio,
        data_sampler=args.train_sampler,
        num_iters=args.num_train_iter,
        num_workers=4 * args.num_workers
    )

    loader_dict['eval'] = get_data_loader(dset_dict['eval'],
        args.eval_batch_size,
        num_workers=args.num_workers,
        drop_last=False
    )


    
    
    ## set DataLoader and ulb_dset on FlexMatch
    model.set_data_loader(loader_dict)

    model.set_dset(ulb_dset)

    # If args.resume, load checkpoints from args.load_path
    if args.resume:
        model.load_model(args.load_path)

    # START TRAINING of flexmatch
    trainer = model.train
    for _ in range(args.epoch):
        trainer(args, logger=logger)


    logging.warning(f"GPU {args.gpu} training is FINISHED")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# main code here begin
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='noisylabels')

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('-sn', '--save_name', type=str, default='flexmatch')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('-o', '--overwrite', action='store_true')
    parser.add_argument('--use_tensorboard', action='store_true', 
        help='Use tensorboard to plot and save curves, otherwise save the curves locally.'
    )

    '''
    Training Configuration of flexmatch
    '''

    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=2 ** 20,
        help='total number of training iterations'
    )
    parser.add_argument('--num_eval_iter', type=int, default=5000,
        help='evaluation frequency'
    )
    parser.add_argument('-nl', '--num_labels', type=int, default=40)
    parser.add_argument('-bsz', '--batch_size', type=int, default=64)
    parser.add_argument('--uratio', type=int, default=7,
        help='the ratio of unlabeled data to labeld data in each mini-batch'
    )
    parser.add_argument('--eval_batch_size', type=int, default=1024,
        help='batch size of evaluation data loader (it does not affect the accuracy)'
    )

    parser.add_argument('--hard_label', type=str2bool, default=True)
    parser.add_argument('--T', type=float, default=0.5)
    parser.add_argument('--p_cutoff', type=float, default=0.95)
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)
    parser.add_argument('--use_DA', type=str2bool, default=False)
    parser.add_argument('-w', '--thresh_warmup', type=str2bool, default=True)

    '''
    Optimizer configurations
    '''
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--amp', type=str2bool, default=False, 
        help='use mixed precision training or not'
    )
    parser.add_argument('--clip', type=float, default=0)
    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--net_from_name', type=str2bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)

    '''
    Data Configurations
    '''

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('-ds', '--dataset', type=str, default='cifar10')
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('-nc', '--num_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)

    ## args for gpu and seed
    parser.add_argument('--seed', default=1, type=int,
        help='seed for initializing training. '
    )
    parser.add_argument('--gpu', default=0, type=int,
        help='GPU id to use.'
    )
    
    # config file (only to change config file to set default value)
    parser.add_argument('--c', type=str, default='')

    # config about noise labels 
    parser.add_argument('--val_ratio', type = float, default = 0)
    parser.add_argument('--noise_type', type = str, 
        help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', 
        default='clean'
    )
    parser.add_argument('--noise_path', type = str, 
        help='path of CIFAR-10_human.pt', default=None
    )
    parser.add_argument('--number_sample', type=int, default=100,
        help='the number of selected samples per class'
    )
    parser.add_argument('--pre_epochs', type=int, default=10,
        help='the epochs of first training step'
    )

    args = parser.parse_args()
    over_write_args_from_file(args, args.c) # read from file

    
    noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
    args.noise_type = noise_type_map[args.noise_type]
    args.name = args.dataset
    # load dataset
    if args.dataset == 'cifar10':
        args.noise_path = './data/CIFAR-10_human.pt'
    elif args.dataset == 'cifar100':
        args.noise_path = './data/CIFAR-100_human.pt'
    main_worker(args)
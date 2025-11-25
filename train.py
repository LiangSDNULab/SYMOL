import argparse
import os

import torch

from model.train_brain import train_10x_Human_breast_cancer_mutifigure, train_10x_Human_breast_cancer

os.environ['R_HOME'] = '/home/lcheng/wangdaoyuan/venv/equation4/lib/R'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Human_breast_cancer', help='should be nanostring, DLPFC, 10x, 10x_brain or merscope,'
                                 'Human_breast_cancer, Mouse_anterior_brain, Mouse_coronal_brain, Human_breast_cancer_cycle Human_colorectal_cance')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for computation (default: cuda if available, otherwise cpu)')#cuda:2
    parser.add_argument('--lr', type=float, default=1e-3)#default=1e-3
    parser.add_argument('--root', type=str, default='./data/nanostring')
    parser.add_argument('--epochs', type=int, default=2000)#default=2000
    parser.add_argument('--id', type=str, default='fov1')
    parser.add_argument('--img_name', type=str, default='F001')
    parser.add_argument('--seed', type=int, default=2024)#1234
    parser.add_argument('--batch_size', type=int, default=512)  # 256 5103
    parser.add_argument('--save_path', type=str, default='./checkpoint/nanostring_final')
    parser.add_argument('--ncluster', type=int, default=8)
    parser.add_argument('--refinement', type=bool, default=False)
    parser.add_argument('--use_gray', type=float, default=0)
    parser.add_argument('--test_only', type=int, default=0)#default=0 #0表示训练，1表示测试
    parser.add_argument('--cluster_method', type=str, default='leiden', help='leiden or mclust')
    parser.add_argument('--cluster_method1', type=str, default='louvain', help='leiden or mclust or louvain')
    parser.add_argument('--cluster_method2', type=str, default='mclust', help='leiden or mclust or louvain')
    parser.add_argument("--temperature_f", default=0.5)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--view_num', type=int, default=2)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--update_interval', default=5, type=int)
    parser.add_argument('--pixel', default=0, type=int)
    parser.add_argument('--rad_cutoff', default=0, type=int)
    parser.add_argument('--initcluster', default="kmeans", type=str)
    parser.add_argument('--sim_batch_size', type=int, default=0, help='Compute NT-Xent loss for a mini-batch or set to 0 to disable')
    parser.add_argument('--net', type=str, default='resnet50', help='resnet50 or resnet101')

    opt = parser.parse_args()

    if opt.dataset == 'nanostring_Lung9_Rep1':
        opt.root = f'/home/lcheng/wangdaoyuan/code/equation5/datasets/{opt.dataset}'
        opt.save_path = f'/home/lcheng/wangdaoyuan/code/equation5/checkpoint/{opt.dataset}'
        opt.rad_cutoff = 80
        opt.net = 'ThreeNet'
        opt.device = torch.device('cpu')
        # train_nanostring(opt, 0)
        # train_nanostring_all(opt, 0)
    elif opt.dataset == 'Human_breast_cancer':
        opt.root = f'/home/lcheng/wangdaoyuan/code/equation5/datasets/{opt.dataset}'
        opt.save_path = f'/home/lcheng/wangdaoyuan/code/equation5/checkpoint/{opt.dataset}'
        opt.device = torch.device('cuda:1')
        opt.ncluster = 20
        opt.rad_cutoff = 400
        opt.refinement = True
        opt.use_gray = 0
        opt.pixel = 66
        opt.epochs = 2000
        opt.net = 'ThreeNet'  # LocalMamba ctranspath gigapath
        train_10x_Human_breast_cancer(opt, 0)

    elif opt.dataset == 'merscope':
        pass
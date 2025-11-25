import ot
import pandas as pd
import os
import scanpy as sc
import numpy as np
import squidpy as sq
import spatialleiden as sl
import cv2
import torch
import torchvision.transforms as transforms
import scipy.sparse as sp
from ot.bregman import empirical_sinkhorn_divergence
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csc_matrix, csr_matrix
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, \
    fowlkes_mallows_score, homogeneity_score
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
import torch.nn.functional as F
from model.model import Pretrain_graph_figure, Pretrain_graph
from model.utils import seed_everything, preprocess, construct_interaction_KNN, construct_interaction, get_feature, \
    get_img, preprocess_adj_sparse, preprocess_adj, construct_interaction_nanostring_KNN, get_brain_img, \
    computeCentroids, labelcontras, regularization_loss, compute_ot_loss
import warnings

warnings.filterwarnings('ignore')

os.environ['R_HOME'] = '/home/lcheng/wangdaoyuan/venv/equation4/lib/R'
os.environ['R_USER'] = '/home/lcheng/wangdaoyuan/venv/equation4/lib/python3.8/site-packages/rpy2'
os.environ['LD_LIBRARY_PATH'] = '/home/lcheng/wangdaoyuan/venv/equation4/lib/R/lib'
os.environ['PYTHONHASHSEED'] = '1234'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def train_10x_Human_breast_cancer(opt, r=0):
    seed_everything(opt.seed)

    #保存数据
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    file_fold = os.path.join(opt.root)
    if opt.dataset =='Mouse_coronal_brain_processed':
        adata = sc.read(os.path.join(file_fold, f'filtered_feature_bc_matrix.h5ad'))
        adata.var_names_make_unique()
    else:
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        adata.var_names_make_unique()

    if 'highly_variable' not in adata.var.keys():
        preprocess(adata)

    if 'adj' not in adata.obsm.keys():
        if opt.dataset in ['Stereo', 'Slide','Mouse_coronal_brain_processed']:#, 'Human_breast_cancer_cycle'
            construct_interaction_KNN(adata, n_neighbors=opt.neighbor)
        else:
            # construct_interaction(adata)
            construct_interaction_nanostring_KNN(adata, rad_cutoff=opt.rad_cutoff)# human: 400

    if 'feat' not in adata.obsm.keys():
        get_feature(adata) # "resnet101" "resnet50"
        get_brain_img(opt, adata)

    # add ground_truth
    df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['ground_truth']
    adata.obs['ground_truth'] = df_meta_layer.values

    if opt.dataset in ['Stereo', 'Slide']:
       #using sparse
       print('Building sparse matrix ...')
       adata.obsm['adj_norm'] = preprocess_adj_sparse(adata.obsm['adj'])
    else:
       # standard version
       adata.obsm['adj_norm'] = preprocess_adj(adata.obsm['adj'])


    in_dim = adata.obsm['feat'].shape[1]
    out_dim = 64
    img_dim = adata.obsm['img_feat2'].shape[1]
    adata.obsm['img_feat'] = adata.obsm['img_feat2']
    # 设置种子
    seed_everything(opt.seed)
    model = Pretrain_graph(in_dim, img_dim, out_dim, opt.ncluster).to(opt.device)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)


    for epoch in tqdm(range(opt.epochs)):
        epoch += 1
        mseloss = 0.0
        loss_nt3 = 0.0
        reg_loss = 0.0
        sinkhorn_divergence = 0.0
        gene = torch.FloatTensor(adata.obsm['feat']).to(opt.device)
        img = torch.FloatTensor(adata.obsm['img_feat']).to(opt.device)
        adj = torch.FloatTensor(adata.obsm['adj_norm']).to(opt.device)
        graph_nei = torch.where(adj > 0, 1, 0).to(opt.device)
        graph_neg = (torch.ones(adj.shape[0], adj.shape[0]).to(opt.device) - graph_nei).to(opt.device)
        gae, gaeout, gz, gzout, iae, iaeout, iz, izout, emb, emb1, emb2, q, q1, q2 = model(gene, img, adj)
        if epoch == 200:
            pca = PCA(n_components=20, random_state=42)
            embedding = pca.fit_transform(emb.detach().cpu().numpy())
            adata.obsm['pred'] = embedding

            ## k-means聚类初始化
            # 设置种子
            seed_everything(opt.seed)
            kmeans = KMeans(n_clusters=opt.ncluster, n_init=100)
            kmeans.fit_predict(embedding)
            model.cluster_model.clusters = torch.nn.Parameter(torch.tensor(kmeans.cluster_centers_).to(opt.device), requires_grad=True)
            y = kmeans.labels_
            y = np.array(y[~pd.isnull(adata.obs['ground_truth'])], dtype=int)
            adata.obs['domain'] = y
            truth = np.array(adata[~pd.isnull(adata.obs['ground_truth'])].obs['ground_truth'], dtype=str)
            ARI = adjusted_rand_score(y, truth)
            print('初始化聚类中心的ARI: %.4f' % ARI)

            emb11 = emb[~pd.isnull(adata.obs['ground_truth'])]
            centers = computeCentroids(emb11.cpu().detach().numpy(), y)
            model.cluster_model.clusters = torch.nn.Parameter(torch.tensor(centers).to(opt.device), requires_grad=True)
            emb1 = emb1[~pd.isnull(adata.obs['ground_truth'])]
            centers1 = computeCentroids(emb1.cpu().detach().numpy(), y)
            model.cluster_model1.clusters = torch.nn.Parameter(torch.tensor(centers1).to(opt.device), requires_grad=True)
            emb2 = emb2[~pd.isnull(adata.obs['ground_truth'])]
            centers2 = computeCentroids(emb2.cpu().detach().numpy(), y)
            model.cluster_model2.clusters = torch.nn.Parameter(torch.tensor(centers2).to(opt.device), requires_grad=True)

        if epoch > 200:
            q = model.cluster_model(emb)
            p = target_distribution(q).detach()
            sinkhorn_divergence = sinkhorn_divergence + empirical_sinkhorn_divergence(q1, p, 1, numIterMax=200)[0] + empirical_sinkhorn_divergence(q2, p, 1, numIterMax=200)[0]
            loss_nt3 = loss_nt3 + labelcontras(p, q1) + labelcontras(p,q2)
            reg_loss = reg_loss + regularization_loss(emb, graph_nei, graph_neg)
        mseloss = mseloss + F.mse_loss(gzout, gene) + F.mse_loss(gaeout, gene) + F.mse_loss(iaeout, img)

        loss = 1 * mseloss + 0.1 * reg_loss + 1 * sinkhorn_divergence + 0.001 * loss_nt3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if (epoch > 100 and epoch % 100 == 0):
                model.eval()
                gae, gaeout, gz, gzout, iae, iaeout, iz, izout, emb, emb1, emb2, q, q1, q2 = model(gene, img, adj)
                adata.obsm['gzout'] = gzout.detach().cpu().numpy()
                # PAC降维
                pca = PCA(n_components=20, random_state=42)
                embedding = pca.fit_transform(emb.detach().cpu().numpy())
                adata.obsm['pred'] = embedding

                if opt.cluster_method2 == 'mclust':
                    # 设置种子
                    seed_everything(opt.seed)
                    mclust_R(adata, opt.ncluster, modelNames='EEE', used_obsm='pred', random_seed=opt.seed)
                    adata.obs['domain'] = adata.obs['mclust']
                    if opt.refinement:
                        new_type = refine_label(adata, 50, key='domain')
                        adata.obs['domain'] = new_type
                    obs_df = adata.obs.dropna()
                    obs_df = obs_df[~pd.isnull(adata.obs['ground_truth'])]
                    ARI3 = np.round(metrics.adjusted_rand_score(obs_df['domain'], obs_df['ground_truth']), 4)
                    NMI = np.round(metrics.normalized_mutual_info_score(obs_df['domain'], obs_df['ground_truth']), 4)
                    AMI = np.round(metrics.adjusted_mutual_info_score(obs_df['domain'], obs_df['ground_truth']), 4)
                    FMI = np.round(metrics.fowlkes_mallows_score(obs_df['domain'], obs_df['ground_truth']), 4)
                    HS = np.round(metrics.homogeneity_score(obs_df['domain'], obs_df['ground_truth']), 4)
                    print('mclust ARI: %.4f' % ARI3)
                    print('mclust NMI: %.4f' % NMI)
                    print('mclust AMI: %.4f' % AMI)
                    print('mclust FMI: %.4f' % FMI)
                    print('mclust HS: %.4f' % HS)


    print("指标保存完成！")

def train_10x_Human_breast_cancer_mutifigure(opt, r=0):
    seed_everything(opt.seed)
    #保存数据
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    file_fold = os.path.join(opt.root)
    if opt.dataset =='Mouse_coronal_brain_processed':
        adata = sc.read(os.path.join(file_fold, f'filtered_feature_bc_matrix.h5ad'))
        adata.var_names_make_unique()
    else:
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        adata.var_names_make_unique()

    if 'highly_variable' not in adata.var.keys():
        preprocess(adata)

    if 'adj' not in adata.obsm.keys():
        if opt.dataset in ['Stereo', 'Slide','Mouse_coronal_brain_processed']:#, 'Human_breast_cancer_cycle'
            construct_interaction_KNN(adata, n_neighbors=opt.neighbor)
        else:
            construct_interaction_nanostring_KNN(adata, rad_cutoff=opt.rad_cutoff)# human: 400

    if 'feat' not in adata.obsm.keys():
        get_feature(adata) # "resnet101" "resnet50"
        get_brain_img(opt, adata)

    # add ground_truth
    df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
    df_meta_layer = df_meta['ground_truth']
    adata.obs['ground_truth'] = df_meta_layer.values

    if opt.dataset in ['Stereo', 'Slide']:
       #using sparse
       print('Building sparse matrix ...')
       adata.obsm['adj_norm'] = preprocess_adj_sparse(adata.obsm['adj'])
    else:
       # standard version
       adata.obsm['adj_norm'] = preprocess_adj(adata.obsm['adj'])

    in_dim = adata.obsm['feat'].shape[1]
    out_dim = 64
    img_dim = 64
    img_dim1 = adata.obsm['img_feat1'].shape[1]
    img_dim2 = adata.obsm['img_feat2'].shape[1]
    img_dim3 = adata.obsm['img_feat3'].shape[1]
    # 设置种子
    seed_everything(opt.seed)
    model = Pretrain_graph_figure(in_dim, img_dim, img_dim1, img_dim2, img_dim3, out_dim, opt.ncluster).to(opt.device)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    for epoch in tqdm(range(opt.epochs)):
        epoch += 1
        mseloss = 0.0
        loss_nt3 = 0.0
        reg_loss = 0.0
        sinkhorn_divergence = 0.0
        gene = torch.FloatTensor(adata.obsm['feat']).to(opt.device)
        img1 = torch.FloatTensor(adata.obsm['img_feat1']).to(opt.device)
        img2 = torch.FloatTensor(adata.obsm['img_feat2']).to(opt.device)
        img3 = torch.FloatTensor(adata.obsm['img_feat3']).to(opt.device)
        adj = torch.FloatTensor(adata.obsm['adj_norm']).to(opt.device)
        graph_nei = torch.where(adj > 0, 1, 0).to(opt.device)
        graph_neg = (torch.ones(adj.shape[0], adj.shape[0]).to(opt.device) - graph_nei).to(opt.device)
        gae, gaeout, gz, gzout, iae, iaeout, iz, izout, emb, emb1, emb2, q, q1, q2, img = model(gene, img1, img2, img3, adj)
        if epoch == 200:
            pca = PCA(n_components=20, random_state=42)
            embedding = pca.fit_transform(emb.detach().cpu().numpy())
            adata.obsm['pred'] = embedding

            ## k-means聚类初始化
            # 设置种子
            seed_everything(opt.seed)
            kmeans = KMeans(n_clusters=opt.ncluster, n_init=100)
            kmeans.fit_predict(embedding)
            model.cluster_model.clusters = torch.nn.Parameter(torch.tensor(kmeans.cluster_centers_).to(opt.device), requires_grad=True)
            y = kmeans.labels_
            y = np.array(y[~pd.isnull(adata.obs['ground_truth'])], dtype=int)
            adata.obs['domain'] = y
            truth = np.array(adata[~pd.isnull(adata.obs['ground_truth'])].obs['ground_truth'], dtype=str)
            ARI = adjusted_rand_score(y, truth)
            print('初始化聚类中心的ARI: %.4f' % ARI)

            emb11 = emb[~pd.isnull(adata.obs['ground_truth'])]
            centers = computeCentroids(emb11.cpu().detach().numpy(), y)
            model.cluster_model.clusters = torch.nn.Parameter(torch.tensor(centers).to(opt.device), requires_grad=True)
            emb1 = emb1[~pd.isnull(adata.obs['ground_truth'])]
            centers1 = computeCentroids(emb1.cpu().detach().numpy(), y)
            model.cluster_model1.clusters = torch.nn.Parameter(torch.tensor(centers1).to(opt.device), requires_grad=True)
            emb2 = emb2[~pd.isnull(adata.obs['ground_truth'])]
            centers2 = computeCentroids(emb2.cpu().detach().numpy(), y)
            model.cluster_model2.clusters = torch.nn.Parameter(torch.tensor(centers2).to(opt.device), requires_grad=True)


        if epoch > 200:
            q = model.cluster_model(emb)
            p = target_distribution(q)#.detach()
            sinkhorn_divergence = sinkhorn_divergence + empirical_sinkhorn_divergence(q1, p, 1, numIterMax=200)[0] + empirical_sinkhorn_divergence(q2, p, 1, numIterMax=200)[0]
            loss_nt3 = loss_nt3 + labelcontras(p, q1) + labelcontras(p,q2)
            reg_loss = reg_loss + regularization_loss(emb, graph_nei, graph_neg)
        mseloss = mseloss + F.mse_loss(gzout, gene) + F.mse_loss(gaeout, gene) + F.mse_loss(iaeout, img)

        loss = 1 * mseloss + 0.1 * reg_loss + 1 * sinkhorn_divergence + 0.01 * loss_nt3# # 参数：10 0.01 0.1 0.001
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            if (epoch > 100 and epoch % 100 == 0):
                model.eval()
                gae, gaeout, gz, gzout, iae, iaeout, iz, izout, emb, emb1, emb2, q, q1, q2, img = model(gene, img1, img2, img3, adj)
                adata.obsm['gzout'] = gzout.detach().cpu().numpy()
                # PAC降维
                pca = PCA(n_components=20, random_state=42)
                embedding = pca.fit_transform(emb.detach().cpu().numpy())
                adata.obsm['pred'] = embedding

                if opt.cluster_method2 == 'mclust':
                    # 设置种子
                    seed_everything(opt.seed)
                    mclust_R(adata, opt.ncluster, modelNames='EEE', used_obsm='pred', random_seed=opt.seed)
                    adata.obs['domain'] = adata.obs['mclust']
                    if opt.refinement:
                        new_type = refine_label(adata, 50, key='domain')
                        adata.obs['domain'] = new_type
                    obs_df = adata.obs.dropna()
                    obs_df = obs_df[~pd.isnull(adata.obs['ground_truth'])]
                    ARI3 = np.round(metrics.adjusted_rand_score(obs_df['domain'], obs_df['ground_truth']), 4)
                    NMI = np.round(metrics.normalized_mutual_info_score(obs_df['domain'], obs_df['ground_truth']), 4)
                    AMI = np.round(metrics.adjusted_mutual_info_score(obs_df['domain'], obs_df['ground_truth']), 4)
                    FMI = np.round(metrics.fowlkes_mallows_score(obs_df['domain'], obs_df['ground_truth']), 4)
                    HS = np.round(metrics.homogeneity_score(obs_df['domain'], obs_df['ground_truth']), 4)
                    print('mclust ARI: %.4f' % ARI3)
                    print('mclust NMI: %.4f' % NMI)
                    print('mclust AMI: %.4f' % AMI)
                    print('mclust FMI: %.4f' % FMI)
                    print('mclust HS: %.4f' % HS)
                    print(len(adata.obs['domain'].unique()))
    print("指标保存完成！")


import torch.nn as nn
import torch.nn.functional as F
L2norm = nn.functional.normalize
def kernel_affinity(z, temperature=0.1, step: int = 3):
    z = L2norm(z)
    G = (2 - 2 * (z @ z.t())).clamp(min=0.)
    G = torch.exp(-G / temperature)
    G = G / G.sum(dim=1, keepdim=True)

    G = torch.matrix_power(G, step)
    alpha = 0.5
    G = torch.eye(G.shape[0]).to(z.device) * alpha + G * (1 - alpha)
    return G

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def clustering(adata, n_clusters=7, radius=50, key='emb', method='mclust', start=0.1, end=3.0, increment=0.01,
               refinement=False):

    pca = PCA(n_components=8, random_state=42)
    embedding = pca.fit_transform(adata.obsm['pred'].copy())
    adata.obsm['emb_pca'] = embedding

    if method == 'mclust':
        adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters)
        adata.obs['domain'] = adata.obs['mclust']
    elif method == 'leiden':
        res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
        res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
        sc.tl.louvain(adata, random_state=0, resolution=res)
        adata.obs['domain'] = adata.obs['louvain']

    if refinement:
        new_type = refine_label(adata, radius, key='domain')
        adata.obs['domain'] = new_type
def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label == 1, "Resolution is not found. Please try bigger range or smaller step!."

    return res

def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    # adata.obs['label_refined'] = np.array(new_type)

    return new_type

def res_search(opt, adata_pred, ncluster, seed, iter=200):
    start = 0;
    end = 3
    i = 0
    while (start < end):
        if i >= iter: return res
        i += 1
        res = (start + end) / 2
        # print(res)
        # 设置种子
        seed_everything(opt.seed)
        sc.tl.leiden(adata_pred, random_state=seed, resolution=res)
        count = len(set(adata_pred.obs['leiden']))
        # print(count)
        if count == ncluster:
            print('find', res)
            return res
        if count > ncluster:
            end = res
        else:
            start = res
    raise NotImplementedError()
def res_search_louvain(opt, adata_pred, ncluster, seed, iter=200):
    start = 0;
    end = 3
    i = 0
    while (start < end):
        if i >= iter: return res
        i += 1
        res = (start + end) / 2
        # print(res)
        # 设置种子
        seed_everything(opt.seed)
        sc.tl.louvain(adata_pred, random_state=seed, resolution=res)
        # sc.tl.leiden(adata_pred, random_state=seed, resolution=res)
        count = len(set(adata_pred.obs['louvain']))
        # print(count)
        if count == ncluster:
            print('find', res)
            return res
        if count > ncluster:
            end = res
        else:
            start = res
    raise NotImplementedError()
def pca(adata, use_reps=None, n_comps=10):
    """Dimension reduction with PCA algorithm"""

    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_comps)
    feat_pca = pca.fit_transform(adata)

    return feat_pca

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()
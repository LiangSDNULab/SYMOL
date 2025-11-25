import cv2
import math
import ot
import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt
import random
import os
os.environ['R_HOME'] = '/home/lcheng/wangdaoyuan/venv/equation5/lib/R'
import torch
import scanpy as sc
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from torch_geometric.data import Data
import torch.backends.cudnn as cudnn
from torchvision.transforms import transforms

cudnn.deterministic = True
cudnn.benchmark = False
from scipy.optimize import linear_sum_assignment
import logging
import anndata2ri
import rpy2
import rpy2.rinterface_lib.callbacks
rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR) # Ignore R warning messages
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri,numpy2ri
from rpy2.robjects.conversion import localconverter
import warnings
warnings.filterwarnings('ignore')
ro.r.source('/home/lcheng/wangdaoyuan/code/equation5/model/batchKL.R')
ro.r.source('/home/lcheng/wangdaoyuan/code/equation5/model/calLISI.R')


##### BatchKL  adata_integraed.obsm["X_emb"]#############
def BatchKL(adata_integrated, batch_column="batch", emb_key="emb"):
    with localconverter(ro.default_converter + pandas2ri.converter):
        meta_data = ro.conversion.py2rpy(adata_integrated.obs)

    from rpy2.robjects import numpy2ri
    numpy2ri.activate()

    embedding = adata_integrated.obsm[emb_key]

    KL = ro.r.BatchKL(meta_data, embedding, n_cells=100, batch=batch_column)
    print("BatchKL=", KL)

    numpy2ri.deactivate()


def LISI(adata_integrated):
    with localconverter(ro.default_converter + pandas2ri.converter):
        meta_data = ro.conversion.py2rpy(adata_integrated.obs)

    from rpy2.robjects import numpy2ri
    numpy2ri.activate()
    embedding = adata_integrated.obsm["emb"]
    lisi = ro.r.CalLISI(embedding, meta_data)
    print("clisi=", lisi[0])
    print("ilisi=", lisi[1])
    numpy2ri.deactivate()
    return lisi

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess(adata):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000, check_values=False)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)

    # sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=5000, check_values=False)
    # sc.pp.normalize_total(adata, target_sum=1e4)
    # sc.pp.log1p(adata)

    # print("start select HVGs")
    # sc.pp.filter_genes(adata, min_cells=100)
    # sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    # adata = adata[:, adata.var['highly_variable']].copy()
    # adata.X = adata.X.toarray()
    # adata.X = adata.X / np.sum(adata.X, axis=1).reshape(-1, 1) * 10000
    # sc.pp.scale(adata, zero_center=False, max_value=10)
    # return adata


def construct_interaction(adata, n_neighbors=3):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm['spatial']

    # calculate distance matrix
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]

    adata.obsm['distance_matrix'] = distance_matrix

    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1

    adata.obsm['graph_neigh'] = interaction

    # transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)

    adata.obsm['adj'] = adj

def construct_interaction_nanostring_KNN(adata, rad_cutoff=80):
    position = adata.obsm['spatial']
    nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(position)
    distances, indices = nbrs.radius_neighbors(position, return_distance=True)
    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

    ## 转变成邻接矩阵， 邻接矩阵开始
    # 初始化邻接矩阵 (如果只需要二进制邻接关系，用 0 填充)
    N = position.shape[0]
    adj_matrix = np.zeros((N, N))
    # 遍历每个样本，填充邻接矩阵
    for i in range(N):
        for j, dist in zip(indices[i], distances[i]):
            if i == j:
                adj_matrix[i, j] = 0
            else:
                adj_matrix[i, j] = 1  # adj_matrix[i, j] = dist 使用距离作为权重，或者用 adj_matrix[i, j] = 1 表示邻接
    # 包括自身graph_neigh
    interaction = adj_matrix + np.eye(adj_matrix.shape[0])

    adj_matrix = adj_matrix + adj_matrix.T
    adj = np.where(adj_matrix > 1, 1, adj_matrix)

    adata.obsm['graph_neigh'] = interaction

    adata.obsm['adj'] = adj
    print('Graph constructed!')

def construct_interaction_KNN(adata, n_neighbors=3):
    position = adata.obsm['spatial']
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(position)
    _, indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1

    adata.obsm['graph_neigh'] = interaction

    # transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)

    adata.obsm['adj'] = adj
    print('Graph constructed!')


def permutation(feature):
    # fix_seed(FLAGS.random_seed)
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]

    return feature_permutated
def get_feature(adata, deconvolution=False):
    if deconvolution:

        adata_Vars = adata
    else:
        adata_Vars = adata[:, adata.var['highly_variable']]

    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
        feat = adata_Vars.X.toarray()[:, ]
    else:
        feat = adata_Vars.X[:, ]

        # data augmentation
    feat_a = permutation(feat)

    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a


def get_brain_img(opt, adata, net="resnet50"):
    # img = cv2.imread(os.path.join(opt.root, 'spatial/full_image.tif'))
    # # 颜色转变为RGB格式的
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # if opt.use_gray:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # transform = transforms.ToTensor()
    # img = transform(img)

    # patchs = []
    # for coor in adata.obsm['spatial']:
    #     py, px = coor
    #     img_p = img[:, px - opt.pixel:px + opt.pixel, py - opt.pixel:py + opt.pixel].flatten()
    #     patchs.append(img_p)
    # patchs = np.stack(patchs)
    # df = pd.DataFrame(patchs, index=adata.obs.index)
    # adata.obsm['img'] = df
    # # anndata中Dataframe数据会报错
    # adata.obsm['img'] = adata.obsm['img'].to_numpy()


    # 读取.npy文件并赋值给 adata.obsm['imgs_feature']
    # feature_filename = f'/home/lcheng/wangdaoyuan/code/equation4/features/Human_breast_cancer/{net}_feature_Human_breast_cancer.npy'
    # feature_filename = f'/home/lcheng/wangdaoyuan/code/equation4/features/Mouse_anterior_brain/{net}_feature_Mouse_anterior_brain.npy'
    # feature_filename = f'/home/lcheng/wangdaoyuan/code/equation4/features/Mouse_coronal_brain/{net}_feature_Mouse_coronal_brain.npy'


    # feature_filename = f'/home/lcheng/wangdaoyuan/code/equation5/features/{opt.dataset}/{opt.net}_feature_{opt.dataset}.npy'
    # if os.path.exists(feature_filename):
    #     features = np.load(feature_filename)
    # else:
    #     print(f"Feature file {feature_filename} does not exist.")
    # features = np.array(features)
    # features = np.vstack(features).astype(np.float64)
    # adata.obsm['img_feat'] = features



    feature_filename = f'/home/lcheng/wangdaoyuan/code/equation5/features/{opt.dataset}/LocalMamba_feature_{opt.dataset}.npy'
    if os.path.exists(feature_filename):
        features = np.load(feature_filename)
    else:
        print(f"Feature file {feature_filename} does not exist.")
    features = np.array(features)
    features = np.vstack(features).astype(np.float64)
    adata.obsm['img_feat1'] = features

    feature_filename = f'/home/lcheng/wangdaoyuan/code/equation5/features/{opt.dataset}/ctranspath_feature_{opt.dataset}.npy'
    if os.path.exists(feature_filename):
        features = np.load(feature_filename)
    else:
        print(f"Feature file {feature_filename} does not exist.")
    features = np.array(features)
    features = np.vstack(features).astype(np.float64)
    adata.obsm['img_feat2'] = features

    feature_filename = f'/home/lcheng/wangdaoyuan/code/equation5/features/{opt.dataset}/gigapath_feature_{opt.dataset}.npy'
    if os.path.exists(feature_filename):
        features = np.load(feature_filename)
    else:
        print(f"Feature file {feature_filename} does not exist.")
    features = np.array(features)
    features = np.vstack(features).astype(np.float64)
    adata.obsm['img_feat3'] = features




def get_img(opt, id, adata, net ="resnet50"):
    # img = cv2.imread(os.path.join(opt.root, id, 'spatial/full_image.tif'))
    # # 颜色转变为RGB格式的
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # if opt.use_gray:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # transform = transforms.ToTensor()
    # img = transform(img)
    #
    # patchs = []
    # for coor in adata.obsm['spatial']:
    #     py, px = coor
    #     img_p = img[:, px - opt.pixel:px + opt.pixel, py - opt.pixel:py + opt.pixel].flatten()
    #     patchs.append(img_p)
    # patchs = np.stack(patchs)
    # df = pd.DataFrame(patchs, index=adata.obs.index)
    # adata.obsm['img'] = df
    # # anndata中Dataframe数据会报错
    # adata.obsm['img'] = adata.obsm['img'].to_numpy()
    # # 读取.npy文件并赋值给 adata.obsm['imgs_feature']

    # feature_filename = f'/home/lcheng/wangdaoyuan/code/eq4/features/DLPFC/resnet50_feature_{id}.npy'
    feature_filename = f'/home/lcheng/wangdaoyuan/code/equation5/features/DLPFC/{net}_feature_{id}.npy'

    if os.path.exists(feature_filename):
        features = np.load(feature_filename)
    else:
        print(f"Feature file {feature_filename} does not exist.")
    features = np.array(features)
    features = np.vstack(features).astype(np.float64)
    adata.obsm['img_feat'] = features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_adj_sparse(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj) + np.eye(adj.shape[0])
    return adj_normalized

def regularization_loss(emb, graph_nei, graph_neg):
    mat = torch.sigmoid(cosine_similarity(emb))  # .cpu()
    # mat = pd.DataFrame(mat.cpu().detach().numpy()).values

    # graph_neg = torch.ones(graph_nei.shape) - graph_nei

    neigh_loss = torch.mul(graph_nei, torch.log(mat)).mean()
    neg_loss = torch.mul(graph_neg, torch.log(1 - mat)).mean()
    pair_loss = -(neigh_loss + neg_loss) / 2
    return pair_loss
def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)

def cosine_similarity(emb):
    mat = torch.matmul(emb, emb.T)
    norm = torch.norm(emb, p=2, dim=1).reshape((emb.shape[0], 1))
    mat = torch.div(mat, torch.matmul(norm, norm.T))
    if torch.any(torch.isnan(mat)):
        mat = _nan2zero(mat)
    mat = mat - torch.diag_embed(torch.diag(mat))
    return mat
def computeCentroids(data, labels):
    n_clusters = len(np.unique(labels))
    return np.array([data[labels == i].mean(0) for i in range(n_clusters)])

def prefilter_genes(adata, min_counts=None, max_counts=None, min_cells=10, max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp = np.asarray([True] * adata.shape[1], dtype=bool)
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, min_cells=min_cells)[0]) if min_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, max_cells=max_cells)[0]) if max_cells is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, min_counts=min_counts)[0]) if min_counts is not None else id_tmp
    id_tmp = np.logical_and(id_tmp,
                            sc.pp.filter_genes(adata.X, max_counts=max_counts)[0]) if max_counts is not None else id_tmp
    adata._inplace_subset_var(id_tmp)
    return id_tmp, adata


def prefilter_specialgenes(adata, Gene1Pattern="ERCC", Gene2Pattern="MT-"):
    id_tmp1 = np.asarray([not str(name).startswith(Gene1Pattern) for name in adata.var_names], dtype=bool)
    id_tmp2 = np.asarray([not str(name).startswith(Gene2Pattern) for name in adata.var_names], dtype=bool)
    id_tmp = np.logical_and(id_tmp1, id_tmp2)
    adata._inplace_subset_var(id_tmp)
    return id_tmp, adata


class InstanceLoss(torch.nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, zi, zj):
        N = 2 * self.batch_size
        z = torch.cat((zi, zj), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_ij = torch.diag(sim, self.batch_size)
        sim_ji = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_ij, sim_ji), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class ClusterLoss(torch.nn.Module):
    def __init__(self, ncluster, temperature):
        super().__init__()
        self.ncluster = ncluster
        self.temperature = temperature

        self.mask = self.mask_correlated_clusters(ncluster)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        self.similarity_f = torch.nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, ncluster):
        N = 2 * ncluster
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(ncluster):
            mask[i, ncluster + 1] = 0
            mask[ncluster + 1, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, ci, cj):
        pi = ci.sum(0).view(-1)
        pi /= pi.sum()
        ne_i = math.log(pi.size(0)) + (pi * torch.log(pi)).sum()
        pj = cj.sum(0).view(-1)
        pj /= pj.sum()
        ne_j = math.log(pj.size(0)) + (pj * torch.log(pj)).sum()

        ne_loss = ne_i + ne_j

        ci = ci.t()
        cj = cj.t()

        N = 2 * self.ncluster
        c = torch.cat((ci, cj), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        # print(sim.shape, self.ncluster)
        sim_ij = torch.diag(sim, self.ncluster)
        sim_ji = torch.diag(sim, -self.ncluster)

        positive_clusters = torch.cat((sim_ij, sim_ji), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss
def mask_correlated_samples(N):
    mask = torch.ones((N, N))
    mask = mask.fill_diagonal_(0)
    for i in range(N//2):
        mask[i, N//2 + i] = 0
        mask[N//2 + i, i] = 0
    mask = mask.bool()
    return mask
def labelcontras( q_i, q_j):
    p_i = q_i.sum(0).view(-1)
    p_i /= p_i.sum()
    ne_i = torch.log(torch.tensor(p_i.size(0))) + (p_i * torch.log(p_i)).sum()
    p_j = q_j.sum(0).view(-1)
    p_j /= p_j.sum()
    ne_j = torch.log(torch.tensor(p_j.size(0))) + (p_j * torch.log(p_j)).sum()
    entropy = ne_i + ne_j

    q_i = q_i.t()
    q_j = q_j.t()
    n_clusters = q_i.size(0)
    N = 2 * n_clusters
    q = torch.cat((q_i, q_j), dim=0)

    sim = nn.CosineSimilarity(dim=2)(q.unsqueeze(1), q.unsqueeze(0)) /1
    sim_i_j = torch.diag(sim, n_clusters)
    sim_j_i = torch.diag(sim, -n_clusters)

    positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    mask = mask_correlated_samples(N)
    negative_clusters = sim[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_clusters.device).long()
    logits = torch.cat((positive_clusters, negative_clusters), dim=1)
    loss = nn.CrossEntropyLoss(reduction="sum")(logits, labels)
    loss /= N
    return loss + entropy


def Transfer_img_Data(adata):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    # print(G_df)
    # exit(0)
    e0 = G_df['Cell1'].to_numpy()
    e1 = G_df['Cell2'].to_numpy()
    edgeList = np.array((e0, e1))

    if type(adata.X) == np.ndarray:
        if 'X_train' in adata.obs.keys():
            X_train_idx = (adata.obs['X_train'].to_numpy() == 1)
            X_test_idx = (adata.obs['X_train'].to_numpy() == 0)
            print(X_train_idx)
            data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X), train_mask=list(X_train_idx),
                val_mask=list(X_test_idx))
            img = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.obsm['imgs']), train_mask=list(X_train_idx),
                val_mask=list(X_test_idx))
        else:
            data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
            img = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.obsm['imgs']))
    else:
        if 'X_train' in adata.obs.keys():
            X_train_idx = (adata.obs['X_train'].to_numpy() == 1)
            X_test_idx = (adata.obs['X_train'].to_numpy() == 0)
            data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()), train_mask=list(X_train_idx),
                val_mask=list(X_test_idx))
            img = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.obsm['imgs'].to_numpy()),
                train_mask=list(X_train_idx), val_mask=list(X_test_idx))
        else:
            data = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
            img = Data(edge_index=torch.LongTensor(np.array(
                [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.obsm['imgs'].to_numpy()))
    return data, img


def Batch_Data(adata, num_batch_x, num_batch_y, spatial_key=['X', 'Y'], plot_Stats=False):
    Sp_df = adata.obs.loc[:, spatial_key].copy()
    Sp_df = np.array(Sp_df)
    batch_x_coor = [np.percentile(Sp_df[:, 0], (1 / num_batch_x) * x * 100) for x in range(num_batch_x + 1)]
    batch_y_coor = [np.percentile(Sp_df[:, 1], (1 / num_batch_y) * x * 100) for x in range(num_batch_y + 1)]

    Batch_list = []
    for it_x in range(num_batch_x):
        for it_y in range(num_batch_y):
            min_x = batch_x_coor[it_x]
            max_x = batch_x_coor[it_x + 1]
            min_y = batch_y_coor[it_y]
            max_y = batch_y_coor[it_y + 1]
            temp_adata = adata.copy()
            temp_adata = temp_adata[temp_adata.obs[spatial_key[0]].map(lambda x: min_x <= x <= max_x)]
            temp_adata = temp_adata[temp_adata.obs[spatial_key[1]].map(lambda y: min_y <= y <= max_y)]
            Batch_list.append(temp_adata)
    if plot_Stats:
        f, ax = plt.subplots(figsize=(1, 3))
        plot_df = pd.DataFrame([x.shape[0] for x in Batch_list], columns=['#spot/batch'])
        sns.boxplot(y='#spot/batch', data=plot_df, ax=ax)
        sns.stripplot(y='#spot/batch', data=plot_df, ax=ax, color='red', size=5)
    return Batch_list


from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix


def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True, use_global=False):
    if verbose:
        print('------Calculating spatial graph...')
    if use_global:
        coor = pd.DataFrame(adata.obsm['spatial_global'])
    else:
        coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))

    ## 转变成邻接矩阵， 邻接矩阵开始
    # 初始化邻接矩阵 (如果只需要二进制邻接关系，用 0 填充)
    N = coor.shape[0]
    adj_matrix = np.zeros((N, N))
    # 遍历每个样本，填充邻接矩阵
    for i in range(N):
        for j, dist in zip(indices[i], distances[i]):
            if i == j:
                adj_matrix[i, j] = 0
            else:
                adj_matrix[i, j] = 1  # adj_matrix[i, j] = dist 使用距离作为权重，或者用 adj_matrix[i, j] = 1 表示邻接
    # 包括自身graph_neigh
    graph_neigh = adj_matrix + np.eye(adj_matrix.shape[0])

    adj_matrix = adj_matrix + adj_matrix.T
    adj_matrix = np.where(adj_matrix > 1, 1, adj_matrix)

    if model == 'KNN':
        graph_neigh, adj_matrix = construct_interaction(adata)

    # 转换为 Pandas DataFrame（如果需要）
    adj_coor = pd.DataFrame(adj_matrix)
    graph_neigh_df = pd.DataFrame(graph_neigh)
    ## 邻接矩阵结束

    # 基于特征构建邻接矩阵
    if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
        adj_feat = kneighbors_graph(adata.X.todense(), 6, mode="connectivity", metric="correlation", include_self=False)
    else:
        adj_feat = kneighbors_graph(adata.X, 6, mode="connectivity", metric="correlation", include_self=False)
    # adj_feat = kneighbors_graph(adata.X.todense(), 6, mode="connectivity", metric="correlation", include_self=False)
    adj_feat = np.array(adj_feat.todense())
    graph_neigh_feat = adj_feat + np.eye(adj_feat.shape[0])
    adj_feat = adj_feat + adj_feat.T
    adj_feat = np.where(adj_feat > 1, 1, adj_feat)
    # 基于特征构建邻接矩阵
    adj_img = kneighbors_graph(adata.obsm['imgs_feature'], 6, mode="connectivity", metric="correlation",
                               include_self=False)
    adj_img = np.array(adj_img.todense())
    graph_neigh_img = adj_img + np.eye(adj_img.shape[0])
    adj_img = adj_img + adj_img.T
    adj_img = np.where(adj_img > 1, 1, adj_img)

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0,]
    # 边的索引
    e0 = Spatial_Net['Cell1'].to_numpy()
    e1 = Spatial_Net['Cell2'].to_numpy()
    edgeList = np.array((e0, e1))
    # adata.obsm['edgeList'] = edgeList
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))

    if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
        P = kernel_affinity(adata.X.todense())
    else:
        P = kernel_affinity(adata.X)

    # P = kernel_affinity(adata.X.todense())
    adata.uns['adj_randwalk'] = P

    adata.uns['adj_coor'] = adj_coor.to_numpy()
    adata.uns['adj_feat'] = adj_feat
    adata.uns['adj_img'] = adj_img
    adata.uns['Spatial_Net'] = Spatial_Net
    adata.uns['graph_neigh_df'] = graph_neigh_df.to_numpy()


def construct_interaction(adata, n_neighbors=3):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm['spatial']

    # calculate distance matrix
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]

    adata.obsm['distance_matrix'] = distance_matrix

    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1

    adata.obsm['graph_neigh'] = interaction

    # transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)

    adata.obsm['adj'] = adj


def compute_ot_loss(P, Q, reg=0.01):
    """
    计算 P 和 Q 之间的最优传输损失 (Sinkhorn 距离)

    :param P: (n, c) 源分布
    :param Q: (m, c) 目标分布
    :param reg: Sinkhorn 正则化参数
    :return: OT 损失值（标量）
    """
    n, c = P.shape
    m, _ = Q.shape

    # 归一化 P 和 Q 使其行和为 1
    P = P / P.sum(dim=1, keepdim=True)  # (n, c)
    Q = Q / Q.sum(dim=1, keepdim=True)  # (m, c)

    # 计算成本矩阵 (n, m)，使用欧几里得距离平方
    cost_matrix = torch.cdist(P, Q, p=2) ** 2  # (n, m)

    # 均匀权重（如果有不同权重，可手动指定）
    P_weights = torch.full((n,), 1/n, dtype=torch.float32)  # (n,)
    Q_weights = torch.full((m,), 1/m, dtype=torch.float32)  # (m,)

    # 计算 Sinkhorn 距离（OT 损失）
    ot_loss = ot.sinkhorn2(P_weights, Q_weights, cost_matrix, reg).sum()

    return ot_loss


from sklearn.decomposition import PCA
import torch.nn as nn

L2norm = nn.functional.normalize


def kernel_affinity(z, temperature=0.1, step: int = 3):
    # # PAC降维
    # pca = PCA(n_components=20, random_state=42)
    # z = pca.fit_transform(z)

    z = L2norm(torch.Tensor(z))
    G = (2 - 2 * (z @ z.t())).clamp(min=0.)
    G = torch.exp(-G / temperature)
    G = G / G.sum(dim=1, keepdim=True)

    G = torch.matrix_power(G, step)
    alpha = 0.2
    G = torch.eye(G.shape[0]) * alpha + G * (1 - alpha)
    return G.numpy()


def Stats_Spatial_Net(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge / adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df / adata.shape[0]
    fig, ax = plt.subplots(figsize=[3, 2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)' % Mean_edge)
    ax.bar(plot_df.index, plot_df)
    plt.close('all')


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='pred', random_seed=0):
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

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames, verbose=False)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def _hungarian_match(flat_preds, flat_target, preds_k, target_k):
    num_samples = flat_target.shape[0]
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))
    for c1 in range(num_k):
        for c2 in range(num_k):
            votes = int(((flat_preds == c1) * (flat_target == c2)).sum())
            num_correct[c1, c2] = votes

    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))
    return res


def distance(X, Y, square=True):
    """
    Compute Euclidean distances between two sets of samples
    Basic framework: pytorch
    :param X: d * n, where d is dimensions and n is number of data points in X
    :param Y: d * m, where m is number of data points in Y
    :param square: whether distances are squared, default value is True
    :return: n * m, distance matrix
    """
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x  # n * 1
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0)
    y = y * y  # m * 1
    y = y.repeat(n, 1)

    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result


def TPL(X, num_neighbors, links=0):
    """
    Solve Problem: Clustering-with-Adaptive-Neighbors(CAN)
    :param X: d * n
    :param num_neighbors:
    :return:
    """
    size = X.shape[1]
    distances = distance(X, X)
    distances = torch.max(distances, torch.t(distances))
    sorted_distances, _ = distances.sort(dim=1)
    top_k = sorted_distances[:, num_neighbors]
    top_k = torch.t(top_k.repeat(size, 1)) + 10 ** -10

    sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors], dim=1)
    sum_top_k = torch.t(sum_top_k.repeat(size, 1))
    sorted_distances = None
    torch.cuda.empty_cache()
    T = top_k - distances
    distances = None
    torch.cuda.empty_cache()
    weights = torch.div(T, num_neighbors * top_k - sum_top_k)
    T = None
    top_k = None
    sum_top_k = None
    torch.cuda.empty_cache()
    weights = weights.relu().cpu()
    if links is not 0:
        links = torch.Tensor(links).cuda(X.device)
        weights += torch.eye(size).cuda(X.device)
        weights += links
        weights /= weights.sum(dim=1).reshape([size, 1])
    torch.cuda.empty_cache()
    raw_weights = weights
    weights = (weights + weights.t()) / 2
    raw_weights = raw_weights.cuda(X.device)
    weights = weights.cuda(X.device)
    return weights, raw_weights

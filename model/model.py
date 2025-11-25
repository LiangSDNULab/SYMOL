import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)# 如果稀疏就用torch.spmm，否则用torch.mm
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, out):
        super(GCN, self).__init__()
        self.gc = GraphConvolution(nfeat, out)
        # self.gc1 = GraphConvolution(128, out)
        # self.act = F.relu

    def forward(self, x, adj):
        x = self.gc(x, adj)
        # x = self.act(x)
        # x = self.gc1(x, adj)
        # x = self.act(x)
        return x

class GraphSAGELayer(nn.Module):
    def __init__(self, in_feats, out_feats, aggregator_type='mean'):
        super(GraphSAGELayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.aggregator_type = aggregator_type
        self.weight = nn.Parameter(torch.FloatTensor(in_feats * 2, out_feats))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        if self.aggregator_type == 'mean':
            aggr_neighbors = torch.spmm(adj, x)
        elif self.aggregator_type == 'max':
            aggr_neighbors = torch.spmm(adj, x).max(dim=1)[0]
        else:
            raise NotImplementedError("Only 'mean' and 'max' aggregators are supported")

        combined = torch.cat([x, aggr_neighbors], dim=1)
        out = torch.mm(combined, self.weight)
        # return nn.PReLU(out)
        return F.relu(out)
        # return out

class GraphSAGE(nn.Module):
    def __init__(self, nfeat, out):
        super(GraphSAGE, self).__init__()
        self.sage = GraphSAGELayer(nfeat, out)
        self.act = F.relu

    def forward(self, x, adj):
        x = self.sage(x, adj)
        return x


class AttentionLayer(nn.Module):
    """\
    Attention layer.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features.

    Returns
    -------
    Aggregated representations and modality weights.

    """

    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1):
        emb = []
        for em in emb1:
             emb.append(torch.unsqueeze(torch.squeeze(em), dim=1))
        self.emb = torch.cat(emb, dim=1)

        self.v = F.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu = torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu) + 1e-6)

        emb_combined = torch.matmul(torch.transpose(self.emb, 1, 2), torch.unsqueeze(self.alpha, -1))

        return torch.squeeze(emb_combined), self.alpha

class ClusteringLayer(nn.Module):

    def __init__(self, n_clusters, latent_dim, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.alpha = alpha
        self.clusters = nn.Parameter(torch.randn(n_clusters, latent_dim), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.clusters, a=math.sqrt(5))

    def forward(self, inputs):
        # 确保 alpha 在相同设备上
        alpha = torch.tensor(self.alpha, device=inputs.device)
        q = 1.0 / (1.0 + (torch.sum(torch.square(torch.unsqueeze(inputs, dim=1) - self.clusters), dim=2) / alpha))
        q = torch.pow(q, (self.alpha + 1.0) / 2.0)
        q = torch.transpose(torch.transpose(q, dim0=0, dim1=1) / torch.sum(q, dim=1), dim0=0, dim1=1)
        return q

class AE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj
class MLP_L(nn.Module):
    def __init__(self, n_mlp, out_dim):
        super(MLP_L, self).__init__()
        self.wl = nn.Linear(n_mlp, out_dim)

    def forward(self, mlp_in):
        weight_output =self.wl(mlp_in)

        return weight_output

class Pretrain_graph(nn.Module):
    def __init__(self, in_dim, img_dim, out_dim, n_clusters, dropout=0.0, act=F.relu):
        super(Pretrain_graph, self).__init__()
        self.dropout = dropout
        self.ae_gene = AE(in_dim, out_dim)
        # Encoder-gene
        self.encoder_gene = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
        # Decoder-gene
        self.decoder_gene = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, in_dim)
        )
        # Encoder-img
        self.encoder_img = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
        # Decoder-img
        self.decoder_img = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, img_dim)
        )
        self.gene_weight1 = Parameter(torch.FloatTensor(in_dim, out_dim))
        self.gene_weight2 = Parameter(torch.FloatTensor(out_dim, in_dim))

        self.ae_img = AE(img_dim, out_dim)
        self.img_weight1 = Parameter(torch.FloatTensor(img_dim, out_dim))
        self.img_weight2 = Parameter(torch.FloatTensor(out_dim, img_dim))

        self.GCN1_gene = GCN(in_dim, out_dim)
        self.GCN2_gene = GCN(out_dim, in_dim)

        self.GSAGE1_gene = GraphSAGE(in_dim, out_dim)
        self.GSAGE2_gene = GraphSAGE(out_dim, in_dim)

        self.GCN1_img = GCN(img_dim, out_dim)
        self.GCN2_img = GCN(out_dim, img_dim)
        self.GSAGE1_img = GraphSAGE(img_dim, out_dim)
        self.GSAGE2_img = GraphSAGE(out_dim, img_dim)

        self.attention1 = AttentionLayer(out_dim, out_dim)
        self.attention2 = AttentionLayer(out_dim, out_dim)
        self.attention = AttentionLayer(out_dim, out_dim)

        self.cluster_model = ClusteringLayer(n_clusters, out_dim)
        self.cluster_model1 = ClusteringLayer(n_clusters, out_dim)
        self.cluster_model2 = ClusteringLayer(n_clusters, out_dim)
        self.reset_parameters()

        self.MLP = nn.Sequential(
            nn.Linear(out_dim*2, out_dim)
        )
        self.MLP1 = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim)
        )

        self.MLP2 = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim)
        )

        self.MLP_L = MLP_L(out_dim, out_dim)
        self.MLP_L1 = MLP_L(out_dim, out_dim)
        self.MLP_L2 = MLP_L(out_dim, out_dim)

        self.MLP_center = nn.Sequential(
            nn.Linear(out_dim, n_clusters)
        )


        self.act = act
        self.sigm = nn.Sigmoid()
        # self.sigm = F.relu

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.gene_weight1)
        torch.nn.init.xavier_uniform_(self.gene_weight2)
        torch.nn.init.xavier_uniform_(self.img_weight1)
        torch.nn.init.xavier_uniform_(self.img_weight2)


    def forward(self, gene, img, adj_coor):

        gene = F.dropout(gene, self.dropout, self.training)
        img = F.dropout(img, self.dropout, self.training)

        gae = self.encoder_gene(gene)
        iae = self.encoder_img(img)

        gaeout = self.decoder_gene(gae)
        iaeout = self.decoder_img(iae)

        gz = self.GSAGE1_gene(gene, adj_coor)
        iz = self.GSAGE1_img(img, adj_coor)
        # gz = self.GCN1_gene(gene, adj_coor)
        # iz = self.GCN1_img(img, adj_coor)

        emb1 = torch.stack([gae, gz], dim=1)
        a1 = self.MLP_L1(emb1)
        emb1 = F.normalize(a1, p=2)
        emb1 = torch.cat((emb1[:, 0].mul(gae), emb1[:, 1].mul(gz)), 1)
        emb1 = self.MLP1(emb1)

        emb2 = torch.stack([iae, iz], dim=1)
        a2 = self.MLP_L2(emb2)
        emb2 = F.normalize(a2, p=2)
        emb2 = torch.cat((emb2[:, 0].mul(iae), emb2[:, 1].mul(iz)), 1)
        emb2 = self.MLP2(emb2)

        emb = torch.stack([emb1, emb2], dim=1)
        a = self.MLP_L(emb)
        emb = F.normalize(a, p=2)
        emb = torch.cat((emb[:, 0].mul(emb1), emb[:, 1].mul(emb2)), 1)
        emb = self.MLP(emb)

        # gzout = self.GCN2_gene(emb, adj_coor)
        # izout = self.GCN2_img(emb, adj_coor)

        gzout = self.GSAGE2_gene(emb, adj_coor)
        izout = self.GSAGE2_img(emb, adj_coor)

        q = self.cluster_model(emb)
        q1 = self.cluster_model1(emb1)
        q2 = self.cluster_model2(emb2)

        return gae, gaeout, gz, gzout, iae, iaeout, iz, izout, emb, emb1, emb2, q, q1, q2



class Pretrain_graph_figure(nn.Module):
    def __init__(self, in_dim, img_dim, img_dim1, img_dim2, img_dim3, out_dim, n_clusters, dropout=0.0, act=F.relu):
        super(Pretrain_graph_figure, self).__init__()
        self.dropout = dropout
        self.ae_gene = AE(in_dim, out_dim)
        # Encoder-gene
        self.encoder_gene = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

        self.encoder_img1 = nn.Sequential(
            nn.Linear(img_dim1, 128)
        )
        self.encoder_img2 = nn.Sequential(
            nn.Linear(img_dim2, 128)
        )
        self.encoder_img3 = nn.Sequential(
            nn.Linear(img_dim3, 128)
        )

        # Decoder-gene
        self.decoder_gene = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, in_dim)
        )
        # Encoder-img
        self.encoder_img = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
        # Decoder-img
        self.decoder_img = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, img_dim)
        )
        self.gene_weight1 = Parameter(torch.FloatTensor(in_dim, out_dim))
        self.gene_weight2 = Parameter(torch.FloatTensor(out_dim, in_dim))

        self.ae_img = AE(img_dim, out_dim)
        self.img_weight1 = Parameter(torch.FloatTensor(img_dim, out_dim))
        self.img_weight2 = Parameter(torch.FloatTensor(out_dim, img_dim))

        self.GCN1_gene = GCN(in_dim, out_dim)
        self.GCN2_gene = GCN(out_dim, in_dim)
        self.GSAGE1_gene = GraphSAGE(in_dim, out_dim)
        self.GSAGE2_gene = GraphSAGE(out_dim, in_dim)

        self.GCN1_img = GCN(img_dim, out_dim)
        self.GCN2_img = GCN(out_dim, img_dim)
        self.GSAGE1_img = GraphSAGE(img_dim, out_dim)
        self.GSAGE2_img = GraphSAGE(out_dim, img_dim)

        self.attention1 = AttentionLayer(out_dim, out_dim)
        self.attention2 = AttentionLayer(out_dim, out_dim)
        self.attention = AttentionLayer(out_dim, out_dim)
        self.attention_img = AttentionLayer(128, 128)

        self.cluster_model = ClusteringLayer(n_clusters, out_dim)
        self.cluster_model1 = ClusteringLayer(n_clusters, out_dim)
        self.cluster_model2 = ClusteringLayer(n_clusters, out_dim)
        self.reset_parameters()

        self.MLP = nn.Sequential(
            nn.Linear(out_dim*2, out_dim)
        )
        self.MLP1 = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim)
        )

        self.MLP2 = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim)
        )
        self.MLP_img = nn.Sequential(
            nn.Linear(128 * 3, 64)
        )
        self.MLP_Limg = MLP_L(128, 128)

        self.MLP_L = MLP_L(out_dim, out_dim)
        self.MLP_L1 = MLP_L(out_dim, out_dim)
        self.MLP_L2 = MLP_L(out_dim, out_dim)

        self.MLP_center = nn.Sequential(
            nn.Linear(out_dim, n_clusters)
        )


        self.act = act
        self.sigm = nn.Sigmoid()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.gene_weight1)
        torch.nn.init.xavier_uniform_(self.gene_weight2)
        torch.nn.init.xavier_uniform_(self.img_weight1)
        torch.nn.init.xavier_uniform_(self.img_weight2)


    def forward(self, gene, img1, img2, img3, adj_coor):

        gene = F.dropout(gene, self.dropout, self.training)
        img1 = F.dropout(img1, self.dropout, self.training)
        img2 = F.dropout(img2, self.dropout, self.training)
        img3 = F.dropout(img3, self.dropout, self.training)
        img_emb1 = self.encoder_img1(img1)
        img_emb2 = self.encoder_img2(img2)
        img_emb3 = self.encoder_img3(img3)

        emb_img = torch.stack([img_emb1, img_emb2, img_emb3], dim=1)
        a_img = self.MLP_Limg(emb_img)
        emb_img = F.normalize(a_img, p=2)  # , eps=1e-12, dim=1
        emb_img = torch.cat((emb_img[:, 0].mul(img_emb1), emb_img[:, 1].mul(img_emb2), emb_img[:, 2].mul(img_emb3)), 1)
        img = self.MLP_img(emb_img)

        gae = self.encoder_gene(gene)
        iae = self.encoder_img(img)

        gaeout = self.decoder_gene(gae)
        iaeout = self.decoder_img(iae)

        gz = self.GSAGE1_gene(gene, adj_coor)
        iz = self.GSAGE1_img(img, adj_coor)

        emb1 = torch.stack([gae, gz], dim=1)
        a1 = self.MLP_L1(emb1)
        emb1 = F.normalize(a1, p=2)
        emb1 = torch.cat((emb1[:, 0].mul(gae), emb1[:, 1].mul(gz)), 1)
        emb1 = self.MLP1(emb1)

        emb2 = torch.stack([iae, iz], dim=1)
        a2 = self.MLP_L2(emb2)
        emb2 = F.normalize(a2, p=2)
        emb2 = torch.cat((emb2[:, 0].mul(iae), emb2[:, 1].mul(iz)), 1)
        emb2 = self.MLP2(emb2)

        emb = torch.stack([emb1, emb2], dim=1)
        a = self.MLP_L(emb)
        emb = F.normalize(a, p=2)
        emb = torch.cat((emb[:, 0].mul(emb1), emb[:, 1].mul(emb2)), 1)
        emb = self.MLP(emb)

        gzout = self.GSAGE2_gene(emb, adj_coor)
        izout = self.GSAGE2_img(emb, adj_coor)

        q = self.cluster_model(emb)
        q1 = self.cluster_model1(emb1)
        q2 = self.cluster_model2(emb2)

        return gae, gaeout, gz, gzout, iae, iaeout, iz, izout, emb, emb1, emb2, q, q1, q2, img


    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def CwCL(self, h_i, h_j, S, temperature_f):
        S_1 = S.repeat(2, 2).to(h_j.device)

        # S_1[S_1 > 0.9] = 1
        # S_1[S_1 < 0.9] = 0
        self.batch_size = S.size(0)
        all_one = torch.ones(self.batch_size*2, self.batch_size*2).to(h_j.device)
        S_2 = all_one - S_1
        # S_2 = all_one

        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)
        self.temperature_f = temperature_f
        sim = torch.matmul(h, h.T) / self.temperature_f
        sim1 = torch.multiply(sim, S_2)

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)

        negative_samples = sim1[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    @staticmethod
    def sim_loss_S(x, x_aug, S, temperature, sym=True):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        all_one = torch.ones(S.size(0), S.size(0)).to(x.device)
        S_1 = all_one - S.to(x.device)
        sim1 = torch.multiply(sim_matrix, S_1)
        mask = torch.ones((batch_size, batch_size)).to(x.device)
        mask = mask.fill_diagonal_(0)
        negative_samples = sim1 * mask

        if sym:
            loss_0 = pos_sim / (negative_samples.sum(dim=0))
            loss_1 = pos_sim / (negative_samples.sum(dim=1))

            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()
            loss = (loss_0 + loss_1) / 2.0
            return loss
        else:
            loss_1 = pos_sim / (negative_samples.sum(dim=1) - pos_sim)
            loss_1 = - torch.log(loss_1).mean()
            return loss_1

    @staticmethod
    def sim_loss(x, x_aug, temperature, sym=True):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        if sym:
            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()
            loss = (loss_0 + loss_1) / 2.0
            return loss
        else:
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss_1 = - torch.log(loss_1).mean()
            return loss_1


import copy
import itertools
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .model import DVCAlignModel
from .utils import create_dictionary_mnn

import torch
import torch.backends.cudnn as cudnn

cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch_geometric.utils.dropout import dropout_edge


class DropFeatures:
    r"""Drops node features with probability p."""

    def __init__(self, p=None, precomputed_weights=True):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
        self.p = p

    def __call__(self, data):
        drop_feat_mask = torch.empty(
            (data.x.size(1),), dtype=torch.float32, device=data.x.device
        ).uniform_(0, 1) < self.p
        data.x[:, drop_feat_mask] = 0
        return data

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)


class DropEdges:
    r"""Drops edges with probability p."""

    def __init__(self, p, force_undirected=False):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
        self.p = p
        self.force_undirected = force_undirected

    def __call__(self, data):
        edge_index = data.edge_index
        edge_index, edge_mask = dropout_edge(
            edge_index, p=self.p, force_undirected=self.force_undirected
        )

        data.edge_index = edge_index
        if getattr(data, 'edge_attr', None) is not None:
            data.edge_attr = data.edge_attr[edge_mask]
        return data

    def __repr__(self):
        return '{}(p={}, force_undirected={})'.format(
            self.__class__.__name__, self.p, self.force_undirected
        )


def get_graph_drop_transform(drop_edge_p, drop_feat_p):
    transforms = list()

    transforms.append(copy.deepcopy)

    if drop_edge_p > 0.:
        transforms.append(DropEdges(drop_edge_p))

    if drop_feat_p > 0.:
        transforms.append(DropFeatures(drop_feat_p))

    return Compose(transforms)


def _masked_graph_inputs(x, edge_index, drop_edge_p=0.0, drop_feat_p=0.0):
    if drop_edge_p <= 0.0 and drop_feat_p <= 0.0:
        return x, edge_index

    graph_data = Data(x=x, edge_index=edge_index)
    transform = get_graph_drop_transform(drop_edge_p=drop_edge_p, drop_feat_p=drop_feat_p)
    masked_graph = transform(graph_data)
    return masked_graph.x, masked_graph.edge_index


class RelationConsistencyLoss(torch.nn.Module):
    def __init__(self):
        super(RelationConsistencyLoss, self).__init__()

    def forward(self, z1, z2):
        normalized_z1 = F.normalize(z1, dim=-1, p=2)
        normalized_z2 = F.normalize(z2, dim=-1, p=2)
        similarity = torch.matmul(normalized_z1, normalized_z2.transpose(1, 0))

        rc_loss = (
            F.mse_loss(similarity, similarity.t()) +
            F.mse_loss(similarity.t(), similarity)
        ) / 2

        return rc_loss, similarity


def _dual_view_forward(model, x, edge_index, edge_index_2=None, drop_edge_p=0.0, drop_feat_p=0.0):
    if edge_index_2 is None:
        edge_index_2 = edge_index

    view1_x, view1_edge_index = _masked_graph_inputs(
        x, edge_index, drop_edge_p=drop_edge_p, drop_feat_p=drop_feat_p
    )
    view2_x, view2_edge_index = _masked_graph_inputs(
        x, edge_index_2, drop_edge_p=drop_edge_p, drop_feat_p=drop_feat_p
    )

    z1, recon1 = model(view1_x, view1_edge_index)
    z2, recon2 = model(view2_x, view2_edge_index)
    return recon1, z1, recon2, z2


def _ensure_batch_name(adata):
    if 'batch_name' in adata.obs.columns:
        return adata

    fallback_columns = [
        'batch',
        'slice_name',
        'slice',
        'section_id',
        'section',
        'sample_name',
        'sample',
    ]
    for column in fallback_columns:
        if column in adata.obs.columns:
            adata.obs['batch_name'] = adata.obs[column].astype(str)
            return adata

    raise KeyError(
        "Could not infer 'batch_name' from adata.obs. "
        f"Available columns: {list(adata.obs.columns)}. "
        "Expected one of: batch_name, batch, slice_name, slice, section_id, section, sample_name, sample."
    )

def Transfer_pytorch_Data(adata):
    if 'Spatial_Net' not in adata.uns:
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    cells = np.asarray(adata.obs_names)
    obs_to_idx = dict(zip(cells, range(cells.shape[0])))

    spatial_net = adata.uns['Spatial_Net'].copy()
    spatial_net = spatial_net[
        spatial_net['Cell1'].isin(obs_to_idx) & spatial_net['Cell2'].isin(obs_to_idx)
    ].copy()
    spatial_net['Cell1'] = spatial_net['Cell1'].map(obs_to_idx)
    spatial_net['Cell2'] = spatial_net['Cell2'].map(obs_to_idx)

    edge_index = np.vstack([
        spatial_net['Cell1'].to_numpy(dtype=np.int64),
        spatial_net['Cell2'].to_numpy(dtype=np.int64),
    ])
    adata.uns['edgeList'] = [edge_index[0], edge_index[1]]

    x = torch.FloatTensor(_to_numpy_matrix(adata.X))
    return Data(edge_index=torch.LongTensor(edge_index), x=x)

def _triplet_warmup_weight(epoch, start_epoch, warmup_epochs, max_weight):
    if epoch < start_epoch:
        return 0.0
    if warmup_epochs <= 0:
        return float(max_weight)

    progress = float(epoch - start_epoch + 1) / float(warmup_epochs)
    progress = min(1.0, max(0.0, progress))
    return float(max_weight) * progress


def _select_closest_mnn_positive(anchor_name, candidate_names, obs_to_idx, embedding):
    if candidate_names is None:
        return None

    if isinstance(candidate_names, (str, bytes)):
        candidate_list = [candidate_names]
    else:
        candidate_list = list(candidate_names)

    if len(candidate_list) == 0:
        return None

    anchor_idx = obs_to_idx.get(anchor_name)
    if anchor_idx is None:
        return candidate_list[0]

    valid_names = []
    valid_indices = []
    for candidate in candidate_list:
        candidate_idx = obs_to_idx.get(candidate)
        if candidate_idx is not None:
            valid_names.append(candidate)
            valid_indices.append(candidate_idx)

    if len(valid_names) == 0:
        return candidate_list[0]

    anchor_vec = embedding[anchor_idx]
    candidate_vec = embedding[np.asarray(valid_indices, dtype=np.int64)]
    dist = np.linalg.norm(candidate_vec - anchor_vec, axis=1)
    closest_idx = int(np.argmin(dist))
    return valid_names[closest_idx]


def _sample_random_negative_from_same_section(anchor_name, section_cell_names):
    if section_cell_names is None or len(section_cell_names) == 0:
        return anchor_name

    if len(section_cell_names) == 1:
        return section_cell_names[0]

    while True:
        negative_name = section_cell_names[np.random.randint(len(section_cell_names))]
        if negative_name != anchor_name:
            return negative_name


def _to_numpy_matrix(matrix):
    if sp.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def _get_spatial_coords(adata):
    if 'spatial' in adata.obsm:
        return np.asarray(adata.obsm['spatial'], dtype=np.float32)

    if {'array_row', 'array_col'}.issubset(adata.obs.columns):
        return adata.obs[['array_row', 'array_col']].to_numpy(dtype=np.float32)

    raise KeyError(
        "Could not infer spatial coordinates from adata. "
        "Expected adata.obsm['spatial'] or obs columns ['array_row', 'array_col']."
    )


def _build_local_expression_edge_list(
    adata,
    batch_name='batch_name',
    candidate_k=40,
    expr_k=12,
    pca_dim=30,
    random_seed=666,
):
    coords = _get_spatial_coords(adata)
    x = _to_numpy_matrix(adata.X).astype(np.float32)
    batch_values = adata.obs[batch_name].astype(str).to_numpy()

    if x.shape[0] <= 1:
        return [np.array([], dtype=np.int64), np.array([], dtype=np.int64)]

    n_components = min(pca_dim, x.shape[1], max(1, x.shape[0] - 1))
    if n_components >= 2 and x.shape[0] > 2:
        pca = PCA(n_components=n_components, random_state=random_seed)
        x_embed = pca.fit_transform(x).astype(np.float32)
    else:
        x_embed = x

    norm = np.linalg.norm(x_embed, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    x_embed = x_embed / norm

    edge_pairs = set()
    for section_id in pd.unique(batch_values):
        section_idx = np.flatnonzero(batch_values == section_id)
        if section_idx.size <= 1:
            continue

        section_coords = coords[section_idx]
        section_expr = x_embed[section_idx]
        local_candidate_k = min(max(expr_k + 1, candidate_k), section_idx.size)
        nbrs = NearestNeighbors(n_neighbors=local_candidate_k, metric='euclidean')
        nbrs.fit(section_coords)
        _, indices = nbrs.kneighbors(section_coords)

        for src_pos in range(section_idx.size):
            cand_rel = indices[src_pos, 1:]
            if cand_rel.size == 0:
                continue

            sim = np.matmul(section_expr[cand_rel], section_expr[src_pos])
            keep = min(max(1, expr_k), cand_rel.size)
            top_rel = cand_rel[np.argsort(-sim)[:keep]]
            src_idx = int(section_idx[src_pos])
            for dst_rel in top_rel:
                dst_idx = int(section_idx[dst_rel])
                if src_idx != dst_idx:
                    edge_pairs.add((src_idx, dst_idx))

    if not edge_pairs:
        return [np.array([], dtype=np.int64), np.array([], dtype=np.int64)]

    edge_pairs = np.asarray(sorted(edge_pairs), dtype=np.int64)
    return [edge_pairs[:, 0], edge_pairs[:, 1]]


def _combine_view_embeddings(model, z_spatial, z_aux, spatial_weight=0.7, use_adaptive_fusion=True, return_gate=False):
    if use_adaptive_fusion and hasattr(model, 'fuse_views'):
        return model.fuse_views(z_spatial, z_aux, fallback_weight=spatial_weight, return_gate=return_gate)

    fused = spatial_weight * z_spatial + (1.0 - spatial_weight) * z_aux
    if return_gate:
        gate = torch.full(
            (z_spatial.shape[0], 1),
            float(spatial_weight),
            dtype=z_spatial.dtype,
            device=z_spatial.device,
        )
        return fused, gate
    return fused


def _compute_clean_dual_embedding(model, x, edge_index_1, edge_index_2, spatial_weight=0.7, use_adaptive_fusion=True,
                                  return_views=False, return_gate=False):
    z1, _ = model(x, edge_index_1)
    z2, _ = model(x, edge_index_2)
    outputs = _combine_view_embeddings(
        model, z1, z2, spatial_weight=spatial_weight, use_adaptive_fusion=use_adaptive_fusion, return_gate=return_gate
    )

    if return_gate:
        fused, gate = outputs
    else:
        fused = outputs
        gate = None

    if return_views and return_gate:
        return fused, z1, z2, gate
    if return_views:
        return fused, z1, z2
    if return_gate:
        return fused, gate
    return fused


def _numpy_cosine_similarity(x, y):
    x_norm = np.linalg.norm(x, axis=1)
    y_norm = np.linalg.norm(y, axis=1)
    denom = np.clip(x_norm * y_norm, a_min=1e-8, a_max=None)
    return np.sum(x * y, axis=1) / denom


def _compute_triplet_confidence(anchor_indices, positive_indices, z_spatial, z_aux,
                                min_weight=0.5, max_weight=1.5):
    if len(anchor_indices) == 0:
        return np.array([], dtype=np.float32)

    sim_spatial = _numpy_cosine_similarity(z_spatial[anchor_indices], z_spatial[positive_indices])
    sim_aux = _numpy_cosine_similarity(z_aux[anchor_indices], z_aux[positive_indices])

    avg_similarity = (sim_spatial + sim_aux + 2.0) / 4.0
    agreement = 1.0 - (np.abs(sim_spatial - sim_aux) / 2.0)
    confidence = np.clip(avg_similarity * agreement, a_min=1e-4, a_max=None)

    confidence = confidence / (confidence.mean() + 1e-8)
    confidence = np.clip(confidence, min_weight, max_weight)
    return confidence.astype(np.float32)


def _weighted_triplet_margin_loss(anchor_arr, positive_arr, negative_arr, margin=1.0, sample_weight=None):
    dist_pos = F.pairwise_distance(anchor_arr, positive_arr, p=2)
    dist_neg = F.pairwise_distance(anchor_arr, negative_arr, p=2)
    loss_per_triplet = F.relu(dist_pos - dist_neg + margin)

    if sample_weight is not None:
        sample_weight = sample_weight.view(-1).to(loss_per_triplet.device, dtype=loss_per_triplet.dtype)
        loss_per_triplet = loss_per_triplet * sample_weight

    return loss_per_triplet.mean()


def _weighted_positive_consistency_loss(anchor_arr, positive_arr, sample_weight=None):
    anchor_arr = F.normalize(anchor_arr, p=2, dim=-1)
    positive_arr = F.normalize(positive_arr, p=2, dim=-1)
    loss_per_pair = 1.0 - torch.sum(anchor_arr * positive_arr, dim=-1)

    if sample_weight is not None:
        sample_weight = sample_weight.view(-1).to(loss_per_pair.device, dtype=loss_per_pair.dtype)
        loss_per_pair = loss_per_pair * sample_weight

    return loss_per_pair.mean()


def _decoder_huber_consistency_loss(recon1, recon2):
    return F.smooth_l1_loss(recon1, recon2)


def _build_chain_consistency_pairs(best_positive_by_anchor, batch_lookup, confidence_by_anchor=None):
    chain_anchor_names = []
    chain_positive_names = []
    chain_weights = []
    used_pairs = set()

    for anchor_name, middle_name in best_positive_by_anchor.items():
        end_name = best_positive_by_anchor.get(middle_name)
        if end_name is None:
            continue

        anchor_batch = batch_lookup.get(anchor_name)
        middle_batch = batch_lookup.get(middle_name)
        end_batch = batch_lookup.get(end_name)
        if anchor_batch is None or middle_batch is None or end_batch is None:
            continue

        if len({anchor_batch, middle_batch, end_batch}) < 3:
            continue
        if anchor_name == end_name:
            continue

        pair_key = (anchor_name, end_name)
        if pair_key in used_pairs:
            continue
        used_pairs.add(pair_key)

        chain_anchor_names.append(anchor_name)
        chain_positive_names.append(end_name)

        if confidence_by_anchor is not None:
            confidence = np.sqrt(
                float(confidence_by_anchor.get(anchor_name, 1.0)) *
                float(confidence_by_anchor.get(middle_name, 1.0))
            )
        else:
            confidence = 1.0
        chain_weights.append(confidence)

    return chain_anchor_names, chain_positive_names, np.asarray(chain_weights, dtype=np.float32)


def train_DVCAlign_pretrain(adata, hidden_dims=[512, 32], n_epochs=3000, lr=0.001, key_added='DVCAlign_pretrain',
                  gradient_clipping=5., weight_decay=0.0001, verbose=True,
                  random_seed=0, save_loss=False, save_reconstrction=False,
                  drop_edge_p=0, drop_feat_p=0, lam_re=1.0, lam_rc=1.0, lam_dec=0,
                  use_aux_expr_graph=True, aux_candidate_k=40, aux_expr_k=20,
                  aux_pca_dim=30, spatial_fusion_weight=0.7, use_adaptive_fusion=True,
                  device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """\
    Training graph attention auto-encoder.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    save_loss
        If True, the final training loss is saved in adata.uns['DVCAlign_pretrain_loss'].
    save_reconstrction
        If True, the reconstructed expression profiles are saved in adata.layers['DVCAlign_pretrain_ReX'].
    drop_edge_p
        Random edge masking probability used during training.
    drop_feat_p
        Random feature masking probability used during training.
    lam_re
        Weight for the dual-view reconstruction loss.
    lam_rc
        Weight for the Relation Consistency Loss between the two views.
    lam_dec
        Weight for the Huber consistency loss between the two decoder outputs.
    use_aux_expr_graph
        Whether to build a local expression graph as the second view.
    aux_candidate_k
        Number of spatially local candidates considered when building the auxiliary expression graph.
    aux_expr_k
        Number of expression-similar neighbors kept from the local candidate set.
    aux_pca_dim
        PCA dimension used when building the auxiliary expression graph.
    spatial_fusion_weight
        Weight of the spatial view when fusing the two view embeddings.
    use_adaptive_fusion
        Whether to use the learnable spot-wise fusion gate instead of a fixed global weight.
    device
        See torch.device.

    Returns
    -------
    AnnData
    """

    # seed_everything()
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    adata = adata.copy()
    adata.X = sp.csr_matrix(adata.X)

    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']].copy()
    else:
        adata_Vars = adata.copy()

    if 'batch_name' not in adata.obs.columns:
        adata.obs['batch_name'] = 'single_section'
    if 'batch_name' not in adata_Vars.obs.columns:
        adata_Vars.obs['batch_name'] = adata.obs.loc[adata_Vars.obs_names, 'batch_name'].astype(str)

    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    _ = Transfer_pytorch_Data(adata_Vars)
    edgeList = adata_Vars.uns['edgeList']
    if use_aux_expr_graph:
        aux_edgeList = _build_local_expression_edge_list(
            adata_Vars,
            batch_name='batch_name',
            candidate_k=aux_candidate_k,
            expr_k=aux_expr_k,
            pca_dim=aux_pca_dim,
            random_seed=random_seed,
        )
        if aux_edgeList[0].size == 0:
            aux_edgeList = edgeList
    else:
        aux_edgeList = edgeList

    data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                prune_edge_index=torch.LongTensor(np.array([aux_edgeList[0], aux_edgeList[1]])),
                x=torch.FloatTensor(_to_numpy_matrix(adata_Vars.X)))

    model = DVCAlignModel(hidden_dims=[data.x.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rc_loss_fn = RelationConsistencyLoss().to(device)

    if verbose:
        print(model)
        if drop_edge_p > 0.0 or drop_feat_p > 0.0:
            print(f'Apply mask augmentation: drop_edge_p={drop_edge_p}, drop_feat_p={drop_feat_p}')
        print(
            f'Use dual-view RC loss: lam_re={lam_re}, lam_rc={lam_rc}, lam_dec={lam_dec}'
        )
        if use_aux_expr_graph:
            print(
                f'Use spatial graph + local-expression graph dual views: '
                f'aux_candidate_k={aux_candidate_k}, aux_expr_k={aux_expr_k}, '
                f'aux_pca_dim={aux_pca_dim}, spatial_fusion_weight={spatial_fusion_weight}'
            )
        else:
            print('Use duplicated spatial graph dual views.')
        print(f'Use adaptive dual-view fusion: {use_adaptive_fusion}')

    loss_list = []
    for epoch in tqdm(range(1, n_epochs + 1)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        recon1, z1, recon2, z2 = _dual_view_forward(
            model, data.x, data.edge_index, data.prune_edge_index,
            drop_edge_p=drop_edge_p, drop_feat_p=drop_feat_p
        )
        recon_loss = (F.mse_loss(data.x, recon1) + F.mse_loss(data.x, recon2)) / 2
        rc_loss, _ = rc_loss_fn(z1, z2)
        dec_loss = _decoder_huber_consistency_loss(recon1, recon2)
        loss = lam_re * recon_loss + lam_rc * rc_loss + lam_dec * dec_loss
        loss_list.append(float(loss.item()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        z = _compute_clean_dual_embedding(
            model, data.x, data.edge_index, data.prune_edge_index,
            spatial_weight=spatial_fusion_weight, use_adaptive_fusion=use_adaptive_fusion
        )
        _, out = model(data.x, data.edge_index)

    embedding = z.to('cpu').detach().numpy()
    adata.obsm[key_added] = embedding
    adata.uns['expr_edgeList'] = aux_edgeList

    if save_loss:
        adata.uns['DVCAlign_pretrain_loss'] = float(loss.item())
    if save_reconstrction:
        ReX = out.to('cpu').detach().numpy()
        ReX[ReX < 0] = 0
        adata.layers['DVCAlign_pretrain_ReX'] = ReX

    return adata

def train_DVCAlign(adata, hidden_dims=[512, 32], n_epochs=1000, lr=0.001, key_added='DVCAlign',
                    gradient_clipping=5., weight_decay=0.0001, margin=1.0, verbose=False,
                    random_seed=666, iter_comb=None, knn_neigh=20,
                    drop_edge_p=0, drop_feat_p=0, lam_re=1.0, lam_rc=1.0, lam_dec=0.05,
                    triplet_warmup_epochs=200, triplet_weight_max=0.7,
                    use_aux_expr_graph=True, aux_candidate_k=40, aux_expr_k=20,
                    aux_pca_dim=30, spatial_fusion_weight=0.5, use_adaptive_fusion=True,
                    use_confidence_triplet=True, triplet_confidence_min=0.3, triplet_confidence_max=1.2,
                    use_chain_consistency=True, chain_consistency_weight=0, chain_warmup_epochs=200,
                    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """\
    Train graph attention auto-encoder and use spot triplets across slices to perform batch correction in the embedding space.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    margin
        Margin is used in triplet loss to enforce the distance between positive and negative pairs.
        Larger values result in more aggressive correction.
    iter_comb
        For multiple slices integration, we perform iterative pairwise integration. iter_comb is used to specify the order of integration.
        For example, (0, 1) means slice 0 will be algined with slice 1 as reference.
    knn_neigh
        The number of nearest neighbors when constructing MNNs. If knn_neigh>1, points in one slice may have multiple MNN points in another slice.
    drop_edge_p
        Random edge masking probability used during training, following Spotscape-style graph augmentation.
    drop_feat_p
        Random feature masking probability used during training, following Spotscape-style graph augmentation.
    lam_re
        Weight for the dual-view reconstruction loss.
    lam_rc
        Weight for the Relation Consistency Loss between the two masked views.
    lam_dec
        Weight for the Huber consistency loss between the two decoder outputs.
    triplet_warmup_epochs
        Number of epochs for linearly warming up triplet loss weight after triplet stage begins.
    triplet_weight_max
        Max weight for triplet loss after warmup.
    use_aux_expr_graph
        Whether to build a local expression graph as the second view.
    aux_candidate_k
        Number of spatially local candidates considered when building the auxiliary expression graph.
    aux_expr_k
        Number of expression-similar neighbors kept from the local candidate set.
    aux_pca_dim
        PCA dimension used when building the auxiliary expression graph.
    spatial_fusion_weight
        Weight of the spatial view when fusing the two view embeddings.
    use_adaptive_fusion
        Whether to use the learnable spot-wise fusion gate instead of a fixed global weight.
    use_confidence_triplet
        Whether to re-weight each triplet by cross-view confidence.
    triplet_confidence_min
        Lower clamp value for confidence-based triplet weights.
    triplet_confidence_max
        Upper clamp value for confidence-based triplet weights.
    use_chain_consistency
        Whether to add a gentle two-hop MNN chain-consistency loss across three distinct slices.
    chain_consistency_weight
        Max weight for the chain-consistency loss after warmup.
    chain_warmup_epochs
        Number of epochs used to warm up the chain-consistency loss after epoch 500.
    device
        See torch.device.

    Returns
    -------
    AnnData
    """

    # seed_everything()
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    adata = _ensure_batch_name(adata)
    section_ids = np.array(adata.obs['batch_name'].unique())
    edgeList = adata.uns['edgeList']
    if use_aux_expr_graph:
        aux_edgeList = _build_local_expression_edge_list(
            adata,
            batch_name='batch_name',
            candidate_k=aux_candidate_k,
            expr_k=aux_expr_k,
            pca_dim=aux_pca_dim,
            random_seed=random_seed,
        )
        if aux_edgeList[0].size == 0:
            aux_edgeList = edgeList
    else:
        aux_edgeList = edgeList

    data = Data(edge_index=torch.LongTensor(np.array([edgeList[0], edgeList[1]])),
                prune_edge_index=torch.LongTensor(np.array([aux_edgeList[0], aux_edgeList[1]])),
                x=torch.FloatTensor(adata.X.todense()))
    data = data.to(device)

    model = DVCAlignModel(hidden_dims=[data.x.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rc_loss_fn = RelationConsistencyLoss().to(device)
    if verbose:
        print(model)
        if drop_edge_p > 0.0 or drop_feat_p > 0.0:
            print(f'Apply mask augmentation: drop_edge_p={drop_edge_p}, drop_feat_p={drop_feat_p}')
        print(
            f'Use dual-view RC loss: lam_re={lam_re}, lam_rc={lam_rc}, lam_dec={lam_dec}, '
            f'triplet_warmup_epochs={triplet_warmup_epochs}, triplet_weight_max={triplet_weight_max}'
        )
        if use_aux_expr_graph:
            print(
                f'Use spatial graph + local-expression graph dual views: '
                f'aux_candidate_k={aux_candidate_k}, aux_expr_k={aux_expr_k}, '
                f'aux_pca_dim={aux_pca_dim}, spatial_fusion_weight={spatial_fusion_weight}'
            )
        else:
            print('Use duplicated spatial graph dual views.')
        print('Triplet positive strategy: pick the closest spot among MNN candidates.')
        print(
            f'Use adaptive dual-view fusion: {use_adaptive_fusion}; '
            f'use confidence-aware triplet: {use_confidence_triplet}; '
            f'use chain consistency: {use_chain_consistency}'
        )

    print('Pretrain DVCAlign encoder...')
    for epoch in tqdm(range(0, 500)):
        model.train()
        optimizer.zero_grad()
        recon1, z1, recon2, z2 = _dual_view_forward(
            model, data.x, data.edge_index, data.prune_edge_index,
            drop_edge_p=drop_edge_p, drop_feat_p=drop_feat_p
        )
        recon_loss = (F.mse_loss(data.x, recon1) + F.mse_loss(data.x, recon2)) / 2
        rc_loss, _ = rc_loss_fn(z1, z2)
        dec_loss = _decoder_huber_consistency_loss(recon1, recon2)
        loss = lam_re * recon_loss + lam_rc * rc_loss + lam_dec * dec_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()

    with torch.no_grad():
        z = _compute_clean_dual_embedding(
            model, data.x, data.edge_index, data.prune_edge_index,
            spatial_weight=spatial_fusion_weight, use_adaptive_fusion=use_adaptive_fusion
        )
    adata.obsm['DVCAlign_pretrain'] = z.cpu().detach().numpy()
    adata.uns['expr_edgeList'] = aux_edgeList

    print('Train DVCAlign...')
    anchor_ind = np.array([], dtype=np.int64)
    positive_ind = np.array([], dtype=np.int64)
    negative_ind = np.array([], dtype=np.int64)
    triplet_confidence = np.array([], dtype=np.float32)
    chain_anchor_ind = np.array([], dtype=np.int64)
    chain_positive_ind = np.array([], dtype=np.int64)
    chain_confidence = np.array([], dtype=np.float32)
    for epoch in tqdm(range(500, n_epochs)):
        if epoch % 100 == 0 or epoch == 500:
            if verbose:
                print('Update spot triplets at epoch ' + str(epoch))
            with torch.no_grad():
                z, z_spatial, z_aux = _compute_clean_dual_embedding(
                    model, data.x, data.edge_index, data.prune_edge_index,
                    spatial_weight=spatial_fusion_weight,
                    use_adaptive_fusion=use_adaptive_fusion,
                    return_views=True,
                )
            adata.obsm['DVCAlign_pretrain'] = z.cpu().detach().numpy()
            embedding_np = adata.obsm['DVCAlign_pretrain']
            z_spatial_np = z_spatial.cpu().detach().numpy()
            z_aux_np = z_aux.cpu().detach().numpy()
            batch_as_dict = dict(zip(list(adata.obs_names), range(0, adata.shape[0])))
            batch_lookup = adata.obs['batch_name'].astype(str).to_dict()

            # If knn_neigh>1, points in one slice may have multiple MNN points in another slice.
            # not all points have MNN achors
            mnn_dict = create_dictionary_mnn(adata, use_rep='DVCAlign_pretrain', batch_name='batch_name', k=knn_neigh,
                                             iter_comb=iter_comb, verbose=0)

            anchor_ind = []
            positive_ind = []
            negative_ind = []
            anchor_name_list = []
            best_positive_by_anchor = {}
            best_positive_distance = {}
            for batch_pair in mnn_dict.keys():  # pairwise compare for multiple batches
                batchname_list = adata.obs['batch_name'][mnn_dict[batch_pair].keys()]
                #             print("before add KNN pairs, len(mnn_dict[batch_pair]):",
                #                   sum(adata_new.obs['batch_name'].isin(batchname_list.unique())), len(mnn_dict[batch_pair]))

                cellname_by_batch_dict = dict()
                for batch_id in range(len(section_ids)):
                    cellname_by_batch_dict[section_ids[batch_id]] = adata.obs_names[
                        adata.obs['batch_name'] == section_ids[batch_id]].values

                anchor_list = []
                positive_list = []
                negative_list = []
                for anchor in mnn_dict[batch_pair].keys():
                    positive_spot = _select_closest_mnn_positive(
                        anchor_name=anchor,
                        candidate_names=mnn_dict[batch_pair][anchor],
                        obs_to_idx=batch_as_dict,
                        embedding=embedding_np,
                    )
                    if positive_spot is None:
                        continue
                    anchor_list.append(anchor)
                    positive_list.append(positive_spot)
                    negative_spot = _sample_random_negative_from_same_section(
                        anchor_name=anchor,
                        section_cell_names=cellname_by_batch_dict[batchname_list[anchor]],
                    )
                    negative_list.append(negative_spot)

                    pair_dist = np.linalg.norm(
                        embedding_np[batch_as_dict[anchor]] - embedding_np[batch_as_dict[positive_spot]]
                    )
                    if anchor not in best_positive_distance or pair_dist < best_positive_distance[anchor]:
                        best_positive_distance[anchor] = pair_dist
                        best_positive_by_anchor[anchor] = positive_spot

                anchor_name_list.extend(anchor_list)
                anchor_ind = np.append(anchor_ind, list(map(lambda _: batch_as_dict[_], anchor_list)))
                positive_ind = np.append(positive_ind, list(map(lambda _: batch_as_dict[_], positive_list)))
                negative_ind = np.append(negative_ind, list(map(lambda _: batch_as_dict[_], negative_list)))

            anchor_ind = np.asarray(anchor_ind, dtype=np.int64)
            positive_ind = np.asarray(positive_ind, dtype=np.int64)
            negative_ind = np.asarray(negative_ind, dtype=np.int64)
            confidence_by_anchor = None
            if use_confidence_triplet and len(anchor_ind) > 0:
                triplet_confidence = _compute_triplet_confidence(
                    anchor_ind,
                    positive_ind,
                    z_spatial_np,
                    z_aux_np,
                    min_weight=triplet_confidence_min,
                    max_weight=triplet_confidence_max,
                )
                confidence_by_anchor = {}
                for anchor_name, confidence in zip(anchor_name_list, triplet_confidence):
                    previous = confidence_by_anchor.get(anchor_name)
                    if previous is None or confidence > previous:
                        confidence_by_anchor[anchor_name] = float(confidence)
            else:
                triplet_confidence = np.ones(len(anchor_ind), dtype=np.float32)

            if use_chain_consistency:
                chain_anchor_names, chain_positive_names, chain_confidence = _build_chain_consistency_pairs(
                    best_positive_by_anchor,
                    batch_lookup=batch_lookup,
                    confidence_by_anchor=confidence_by_anchor,
                )
                chain_anchor_ind = np.asarray(
                    [batch_as_dict[name] for name in chain_anchor_names], dtype=np.int64
                ) if len(chain_anchor_names) > 0 else np.array([], dtype=np.int64)
                chain_positive_ind = np.asarray(
                    [batch_as_dict[name] for name in chain_positive_names], dtype=np.int64
                ) if len(chain_positive_names) > 0 else np.array([], dtype=np.int64)
                if chain_confidence.size > 0:
                    chain_confidence = np.clip(
                        chain_confidence, triplet_confidence_min, triplet_confidence_max
                    )
                else:
                    chain_confidence = np.ones(len(chain_anchor_ind), dtype=np.float32)
            else:
                chain_anchor_ind = np.array([], dtype=np.int64)
                chain_positive_ind = np.array([], dtype=np.int64)
                chain_confidence = np.array([], dtype=np.float32)

        model.train()
        optimizer.zero_grad()
        recon1, z1, recon2, z2 = _dual_view_forward(
            model, data.x, data.edge_index, data.prune_edge_index,
            drop_edge_p=drop_edge_p, drop_feat_p=drop_feat_p
        )
        recon_loss = (F.mse_loss(data.x, recon1) + F.mse_loss(data.x, recon2)) / 2
        rc_loss, _ = rc_loss_fn(z1, z2)
        dec_loss = _decoder_huber_consistency_loss(recon1, recon2)
        z = _combine_view_embeddings(
            model, z1, z2, spatial_weight=spatial_fusion_weight, use_adaptive_fusion=use_adaptive_fusion
        )
        tri_weight = _triplet_warmup_weight(
            epoch=epoch,
            start_epoch=500,
            warmup_epochs=triplet_warmup_epochs,
            max_weight=triplet_weight_max,
        )
        chain_weight = _triplet_warmup_weight(
            epoch=epoch,
            start_epoch=500,
            warmup_epochs=chain_warmup_epochs,
            max_weight=chain_consistency_weight,
        )

        if len(anchor_ind) > 0 and tri_weight > 0.0:
            anchor_arr = z[anchor_ind,]
            positive_arr = z[positive_ind,]
            negative_arr = z[negative_ind,]
            sample_weight = None
            if use_confidence_triplet and len(triplet_confidence) == len(anchor_ind):
                sample_weight = torch.from_numpy(triplet_confidence).to(device=device, dtype=z.dtype)
            tri_output = _weighted_triplet_margin_loss(
                anchor_arr,
                positive_arr,
                negative_arr,
                margin=margin,
                sample_weight=sample_weight,
            )
        else:
            tri_output = torch.tensor(0.0, device=device)

        if use_chain_consistency and len(chain_anchor_ind) > 0 and chain_weight > 0.0:
            chain_anchor_arr = z[chain_anchor_ind,]
            chain_positive_arr = z[chain_positive_ind,]
            chain_sample_weight = None
            if len(chain_confidence) == len(chain_anchor_ind):
                chain_sample_weight = torch.from_numpy(chain_confidence).to(device=device, dtype=z.dtype)
            chain_output = _weighted_positive_consistency_loss(
                chain_anchor_arr,
                chain_positive_arr,
                sample_weight=chain_sample_weight,
            )
        else:
            chain_output = torch.tensor(0.0, device=device)

        loss = (
            lam_re * recon_loss
            + lam_rc * rc_loss
            + lam_dec * dec_loss
            + tri_weight * tri_output
            + chain_weight * chain_output
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

    #
    model.eval()
    with torch.no_grad():
        z = _compute_clean_dual_embedding(
            model, data.x, data.edge_index, data.prune_edge_index,
            spatial_weight=spatial_fusion_weight, use_adaptive_fusion=use_adaptive_fusion
        )
    adata.obsm[key_added] = z.cpu().detach().numpy()
    return adata


def train_DVCAlign_subgraph(adata, hidden_dims=[512, 32], n_epochs=1000, lr=0.001, key_added='DVCAlign',
                             gradient_clipping=5., weight_decay=0.0001, margin=1.0, verbose=False,
                             random_seed=666, iter_comb=None, knn_neigh=100, Batch_list=None,
                             drop_edge_p=0.0, drop_feat_p=0.0, lam_re=1.0, lam_rc=1.0, lam_dec=0.05,
                             triplet_warmup_epochs=200, triplet_weight_max=0.1,
                             use_aux_expr_graph=True, aux_candidate_k=40, aux_expr_k=20,
                             aux_pca_dim=30, spatial_fusion_weight=0.7, use_adaptive_fusion=True,
                             use_confidence_triplet=True, triplet_confidence_min=0.5, triplet_confidence_max=1.5,
                             use_chain_consistency=True, chain_consistency_weight=0, chain_warmup_epochs=250,
                             rc_sample_size=2048,
                             device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """\
    Train graph attention auto-encoder and use spot triplets across slices to perform batch correction in the embedding space.
    To deal with large-scale data with multiple slices and reduce GPU memory usage, each slice is considered as a subgraph for training.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    margin
        Margin is used in triplet loss to enforce the distance between positive and negative pairs.
        Larger values result in more aggressive correction.
    iter_comb
        For multiple slices integration, we perform iterative pairwise integration. iter_comb is used to specify the order of integration.
        For example, (0, 1) means slice 0 will be algined with slice 1 as reference.
    knn_neigh
        The number of nearest neighbors when constructing MNNs. If knn_neigh>1, points in one slice may have multiple MNN points in another slice.
    drop_edge_p
        Random edge masking probability used during training, following Spotscape-style graph augmentation.
    drop_feat_p
        Random feature masking probability used during training, following Spotscape-style graph augmentation.
    lam_re
        Weight for the dual-view reconstruction loss.
    lam_rc
        Weight for the Relation Consistency Loss between the two masked views.
    lam_dec
        Weight for the Huber consistency loss between the two decoder outputs.
    triplet_warmup_epochs
        Number of epochs for linearly warming up triplet loss weight after triplet stage begins.
    triplet_weight_max
        Max weight for triplet loss after warmup.
    use_aux_expr_graph
        Whether to build a local expression graph as the second view for each slice subgraph.
    aux_candidate_k
        Number of spatially local candidates considered when building the auxiliary expression graph.
    aux_expr_k
        Number of expression-similar neighbors kept from the local candidate set.
    aux_pca_dim
        PCA dimension used when building the auxiliary expression graph.
    spatial_fusion_weight
        Weight of the spatial view when fusing the two view embeddings.
    use_adaptive_fusion
        Whether to use the learnable spot-wise fusion gate instead of a fixed global weight.
    use_confidence_triplet
        Whether to re-weight each triplet by cross-view confidence.
    triplet_confidence_min
        Lower clamp value for confidence-based triplet weights.
    triplet_confidence_max
        Upper clamp value for confidence-based triplet weights.
    use_chain_consistency
        Whether to add a gentle two-hop MNN chain-consistency loss across three distinct slices.
    chain_consistency_weight
        Max weight for the chain-consistency loss after warmup.
    chain_warmup_epochs
        Number of epochs used to warm up the chain-consistency loss after epoch 500.
    rc_sample_size
        Number of spots sampled to estimate the relation consistency loss. If None or not smaller
        than the current batch size, the full pairwise relation consistency loss is used.
    device
        See torch.device.

    Returns
    -------
    AnnData
    """

    # seed_everything()
    seed = random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    def _compute_subgraph_rc_loss(z1, z2):
        if lam_rc <= 0:
            return torch.zeros((), device=z1.device, dtype=z1.dtype)

        sample_size = rc_sample_size
        if sample_size is None or sample_size <= 0 or sample_size >= z1.size(0):
            rc_loss, _ = rc_loss_fn(z1, z2)
            return rc_loss

        sample_idx = torch.randperm(z1.size(0), device=z1.device)[:sample_size]
        rc_loss, _ = rc_loss_fn(
            z1.index_select(0, sample_idx),
            z2.index_select(0, sample_idx),
        )
        return rc_loss

    def _concat_slice_pair(slice_i, slice_j):
        slice_a = Batch_list[slice_i][:, comm_gene].copy()
        slice_b = Batch_list[slice_j][:, comm_gene].copy()
        pair = ad.concat(
            [slice_a, slice_b],
            label='batch_name',
            keys=[section_ids[slice_i], section_ids[slice_j]],
            merge='same',
        )
        pair.obs['batch_name'] = pair.obs['batch_name'].astype(str)
        return pair

    def _merge_two_edge_lists(edge_a, edge_b, offset):
        merged_src = np.append(edge_a[0], edge_b[0] + offset).astype(np.int64, copy=False)
        merged_dst = np.append(edge_a[1], edge_b[1] + offset).astype(np.int64, copy=False)
        return [merged_src, merged_dst]

    def _build_pair_training_data(pair_key, triplet_payload=None, chain_payload=None):
        slice_i, slice_j = pair_key
        batch_pair = _concat_slice_pair(slice_i, slice_j)
        n_slice_i = Batch_list[slice_i].shape[0]
        spatial_edges = _merge_two_edge_lists(slice_spatial_edges[slice_i], slice_spatial_edges[slice_j], n_slice_i)
        aux_edges = _merge_two_edge_lists(slice_aux_edges[slice_i], slice_aux_edges[slice_j], n_slice_i)
        batch_as_dict = dict(zip(list(batch_pair.obs_names), range(batch_pair.shape[0])))

        triplet_payload = triplet_payload or {}
        chain_payload = chain_payload or {}

        anchor_names = triplet_payload.get('anchor_names', [])
        positive_names = triplet_payload.get('positive_names', [])
        negative_names = triplet_payload.get('negative_names', [])
        triplet_weight = triplet_payload.get('triplet_weight', np.array([], dtype=np.float32))

        chain_anchor_names = chain_payload.get('anchor_names', [])
        chain_positive_names = chain_payload.get('positive_names', [])
        chain_weight = chain_payload.get('chain_weight', np.array([], dtype=np.float32))

        return Data(
            edge_index=torch.LongTensor(np.array([spatial_edges[0], spatial_edges[1]])),
            prune_edge_index=torch.LongTensor(np.array([aux_edges[0], aux_edges[1]])),
            anchor_ind=torch.LongTensor(np.asarray([batch_as_dict[name] for name in anchor_names], dtype=np.int64)),
            positive_ind=torch.LongTensor(np.asarray([batch_as_dict[name] for name in positive_names], dtype=np.int64)),
            negative_ind=torch.LongTensor(np.asarray([batch_as_dict[name] for name in negative_names], dtype=np.int64)),
            triplet_weight=torch.FloatTensor(
                triplet_weight if len(triplet_weight) > 0 else np.array([], dtype=np.float32)
            ),
            chain_anchor_ind=torch.LongTensor(
                np.asarray([batch_as_dict[name] for name in chain_anchor_names], dtype=np.int64)
            ),
            chain_positive_ind=torch.LongTensor(
                np.asarray([batch_as_dict[name] for name in chain_positive_names], dtype=np.int64)
            ),
            chain_weight=torch.FloatTensor(
                chain_weight if len(chain_weight) > 0 else np.array([], dtype=np.float32)
            ),
            x=batch_pair.X,
        )

    def _extract_dual_embeddings_cpu():
        model_cpu = model.cpu()
        z_list = []
        z_spatial_list = []
        z_aux_list = []
        with torch.no_grad():
            for batch in data_list:
                z_spatial, _ = model_cpu(batch.x, batch.edge_index)
                z_aux, _ = model_cpu(batch.x, batch.prune_edge_index)
                z = _combine_view_embeddings(
                    model_cpu,
                    z_spatial,
                    z_aux,
                    spatial_weight=spatial_fusion_weight,
                    use_adaptive_fusion=use_adaptive_fusion,
                )
                z_list.append(z.cpu().detach().numpy())
                z_spatial_list.append(z_spatial.cpu().detach().numpy())
                z_aux_list.append(z_aux.cpu().detach().numpy())
        return np.concatenate(z_list, axis=0), np.concatenate(z_spatial_list, axis=0), np.concatenate(z_aux_list, axis=0)

    adata = _ensure_batch_name(adata)
    section_ids = np.array(adata.obs['batch_name'].unique())
    global_obs_to_idx = dict(zip(list(adata.obs_names), range(adata.shape[0])))
    section_to_idx = {str(section_id): idx for idx, section_id in enumerate(section_ids)}

    if Batch_list is None or len(Batch_list) == 0:
        raise ValueError("Batch_list must be provided for train_DVCAlign_subgraph.")

    if iter_comb is None:
        iter_comb = list(itertools.combinations(range(len(section_ids)), 2))
    iter_comb = [tuple(map(int, comb)) for comb in iter_comb]

    comm_gene = adata.var_names
    data_list = []
    slice_spatial_edges = []
    slice_aux_edges = []
    for batch_idx, adata_tmp in enumerate(Batch_list):
        adata_tmp = adata_tmp[:, comm_gene].copy()
        if 'batch_name' not in adata_tmp.obs.columns:
            adata_tmp.obs['batch_name'] = str(section_ids[batch_idx])
        adata_tmp = _ensure_batch_name(adata_tmp)
        edge_index = np.nonzero(adata_tmp.uns['adj'])
        spatial_edge_list = [
            np.asarray(edge_index[0], dtype=np.int64),
            np.asarray(edge_index[1], dtype=np.int64),
        ]
        if use_aux_expr_graph:
            aux_edge_list = _build_local_expression_edge_list(
                adata_tmp,
                batch_name='batch_name',
                candidate_k=aux_candidate_k,
                expr_k=aux_expr_k,
                pca_dim=aux_pca_dim,
                random_seed=random_seed,
            )
            if aux_edge_list[0].size == 0:
                aux_edge_list = spatial_edge_list
        else:
            aux_edge_list = spatial_edge_list

        slice_spatial_edges.append(spatial_edge_list)
        slice_aux_edges.append(aux_edge_list)
        data_list.append(
            Data(
                edge_index=torch.LongTensor(np.array([spatial_edge_list[0], spatial_edge_list[1]])),
                prune_edge_index=torch.LongTensor(np.array([aux_edge_list[0], aux_edge_list[1]])),
                x=torch.FloatTensor(_to_numpy_matrix(adata_tmp.X)),
            )
        )

    loader = DataLoader(data_list, batch_size=1, shuffle=True)

    model = DVCAlignModel(hidden_dims=[adata.X.shape[1], hidden_dims[0], hidden_dims[1]]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rc_loss_fn = RelationConsistencyLoss().to(device)
    if verbose:
        print(model)
        if drop_edge_p > 0.0 or drop_feat_p > 0.0:
            print(f'Apply mask augmentation: drop_edge_p={drop_edge_p}, drop_feat_p={drop_feat_p}')
        print(
            f'Use dual-view RC loss: lam_re={lam_re}, lam_rc={lam_rc}, lam_dec={lam_dec}, '
            f'triplet_warmup_epochs={triplet_warmup_epochs}, triplet_weight_max={triplet_weight_max}, '
            f'chain_consistency_weight={chain_consistency_weight}, chain_warmup_epochs={chain_warmup_epochs}, '
            f'rc_sample_size={rc_sample_size}'
        )
        if use_aux_expr_graph:
            print(
                f'Use spatial graph + local-expression graph dual views: '
                f'aux_candidate_k={aux_candidate_k}, aux_expr_k={aux_expr_k}, '
                f'aux_pca_dim={aux_pca_dim}, spatial_fusion_weight={spatial_fusion_weight}'
            )
        else:
            print('Use duplicated spatial graph dual views.')
        print('Triplet positive strategy: pick the closest spot among MNN candidates.')
        print(
            f'Use adaptive dual-view fusion: {use_adaptive_fusion}; '
            f'use confidence-aware triplet: {use_confidence_triplet}; '
            f'use chain consistency: {use_chain_consistency}'
        )

    print('Pretrain DVCAlign encoder...')
    for epoch in tqdm(range(0, 500)):
        for batch in loader:
            model.train()
            optimizer.zero_grad()
            batch = batch.to(device)
            recon1, z1, recon2, z2 = _dual_view_forward(
                model, batch.x, batch.edge_index, batch.prune_edge_index,
                drop_edge_p=drop_edge_p, drop_feat_p=drop_feat_p
            )
            recon_loss = (F.mse_loss(batch.x, recon1) + F.mse_loss(batch.x, recon2)) / 2
            rc_loss = _compute_subgraph_rc_loss(z1, z2)
            dec_loss = _decoder_huber_consistency_loss(recon1, recon2)
            loss = lam_re * recon_loss + lam_rc * rc_loss + lam_dec * dec_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

    adata.obsm['DVCAlign_pretrain'], _, _ = _extract_dual_embeddings_cpu()
    model = model.to(device)

    print('Train DVCAlign...')
    for epoch in tqdm(range(500, n_epochs)):
        if epoch % 100 == 0 or epoch == 500:
            if verbose:
                print('Update spot triplets at epoch ' + str(epoch))

            adata.obsm['DVCAlign_pretrain'], embedding_spatial_np, embedding_aux_np = _extract_dual_embeddings_cpu()
            model = model.to(device)

            best_positive_by_anchor = {}
            best_positive_distance = {}
            confidence_by_anchor = {}
            triplet_payload_by_pair = {}
            for comb in iter_comb:
                i, j = sorted((comb[0], comb[1]))
                pair_key = (i, j)
                batch_pair = _concat_slice_pair(i, j)
                batch_pair.obsm['DVCAlign_pretrain'] = adata.obsm['DVCAlign_pretrain'][
                    np.asarray([global_obs_to_idx[name] for name in batch_pair.obs_names], dtype=np.int64)
                ]
                mnn_dict = create_dictionary_mnn(
                    batch_pair,
                    use_rep='DVCAlign_pretrain',
                    batch_name='batch_name',
                    k=knn_neigh,
                    iter_comb=None,
                    verbose=0,
                )
                embedding_np = np.asarray(batch_pair.obsm['DVCAlign_pretrain'])

                batchname_list = batch_pair.obs['batch_name'].astype(str)
                cellname_by_batch_dict = {}
                for batch_id in pair_key:
                    cellname_by_batch_dict[section_ids[batch_id]] = batch_pair.obs_names[
                        batchname_list == section_ids[batch_id]
                    ].values

                anchor_list = []
                positive_list = []
                negative_list = []
                batch_as_dict = dict(zip(list(batch_pair.obs_names), range(0, batch_pair.shape[0])))
                anchor_name_list = []
                for batch_pair_name in mnn_dict.keys():  # pairwise compare for multiple batches
                    for anchor in mnn_dict[batch_pair_name].keys():
                        positive_spot = _select_closest_mnn_positive(
                            anchor_name=anchor,
                            candidate_names=mnn_dict[batch_pair_name][anchor],
                            obs_to_idx=batch_as_dict,
                            embedding=embedding_np,
                        )
                        if positive_spot is None:
                            continue
                        anchor_list.append(anchor)
                        anchor_name_list.append(anchor)
                        positive_list.append(positive_spot)
                        negative_list.append(
                            _sample_random_negative_from_same_section(
                                anchor_name=anchor,
                                section_cell_names=cellname_by_batch_dict[batchname_list[anchor]],
                            )
                        )
                        pair_dist = np.linalg.norm(
                            embedding_np[batch_as_dict[anchor]] - embedding_np[batch_as_dict[positive_spot]]
                        )
                        if anchor not in best_positive_distance or pair_dist < best_positive_distance[anchor]:
                            best_positive_distance[anchor] = pair_dist
                            best_positive_by_anchor[anchor] = positive_spot

                anchor_ind = np.asarray(list(map(lambda _: batch_as_dict[_], anchor_list)), dtype=np.int64)
                positive_ind = np.asarray(list(map(lambda _: batch_as_dict[_], positive_list)), dtype=np.int64)
                negative_ind = np.asarray(list(map(lambda _: batch_as_dict[_], negative_list)), dtype=np.int64)
                if use_confidence_triplet and len(anchor_ind) > 0:
                    pair_global_idx = np.asarray([global_obs_to_idx[name] for name in batch_pair.obs_names], dtype=np.int64)
                    confidence = _compute_triplet_confidence(
                        anchor_ind,
                        positive_ind,
                        embedding_spatial_np[pair_global_idx],
                        embedding_aux_np[pair_global_idx],
                        min_weight=triplet_confidence_min,
                        max_weight=triplet_confidence_max,
                    )
                    for anchor_name, conf in zip(anchor_name_list, confidence):
                        previous = confidence_by_anchor.get(anchor_name)
                        if previous is None or conf > previous:
                            confidence_by_anchor[anchor_name] = float(conf)
                else:
                    confidence = np.ones(len(anchor_ind), dtype=np.float32)

                triplet_payload_by_pair[pair_key] = {
                    'anchor_names': anchor_list,
                    'positive_names': positive_list,
                    'negative_names': negative_list,
                    'triplet_weight': confidence.astype(np.float32),
                }

            chain_payload_by_pair = {}
            if use_chain_consistency:
                batch_lookup = adata.obs['batch_name'].astype(str).to_dict()
                chain_anchor_names, chain_positive_names, chain_confidence = _build_chain_consistency_pairs(
                    best_positive_by_anchor,
                    batch_lookup=batch_lookup,
                    confidence_by_anchor=confidence_by_anchor if len(confidence_by_anchor) > 0 else None,
                )
                if chain_confidence.size > 0:
                    chain_confidence = np.clip(
                        chain_confidence,
                        triplet_confidence_min,
                        triplet_confidence_max,
                    )

                for anchor_name, positive_name, chain_conf in zip(
                    chain_anchor_names, chain_positive_names, chain_confidence
                ):
                    pair_key = tuple(
                        sorted(
                            (
                                section_to_idx[batch_lookup[anchor_name]],
                                section_to_idx[batch_lookup[positive_name]],
                            )
                        )
                    )
                    payload = chain_payload_by_pair.setdefault(
                        pair_key,
                        {'anchor_names': [], 'positive_names': [], 'chain_weight': []},
                    )
                    payload['anchor_names'].append(anchor_name)
                    payload['positive_names'].append(positive_name)
                    payload['chain_weight'].append(float(chain_conf))

            pair_keys = sorted(set(triplet_payload_by_pair.keys()) | set(chain_payload_by_pair.keys()))
            pair_data_list = [
                _build_pair_training_data(
                    pair_key,
                    triplet_payload=triplet_payload_by_pair.get(pair_key),
                    chain_payload=chain_payload_by_pair.get(pair_key),
                )
                for pair_key in pair_keys
            ]
            pair_loader = DataLoader(pair_data_list, batch_size=1, shuffle=True)

        for batch in pair_loader:
            model.train()
            optimizer.zero_grad()

            batch.x = torch.FloatTensor(batch.x[0].todense())
            batch = batch.to(device)
            recon1, z1, recon2, z2 = _dual_view_forward(
                model, batch.x, batch.edge_index, batch.prune_edge_index,
                drop_edge_p=drop_edge_p, drop_feat_p=drop_feat_p
            )
            recon_loss = (F.mse_loss(batch.x, recon1) + F.mse_loss(batch.x, recon2)) / 2
            rc_loss = _compute_subgraph_rc_loss(z1, z2)
            dec_loss = _decoder_huber_consistency_loss(recon1, recon2)
            z = _combine_view_embeddings(
                model, z1, z2, spatial_weight=spatial_fusion_weight, use_adaptive_fusion=use_adaptive_fusion
            )
            tri_weight = _triplet_warmup_weight(
                epoch=epoch,
                start_epoch=500,
                warmup_epochs=triplet_warmup_epochs,
                max_weight=triplet_weight_max,
            )
            chain_weight = _triplet_warmup_weight(
                epoch=epoch,
                start_epoch=500,
                warmup_epochs=chain_warmup_epochs,
                max_weight=chain_consistency_weight,
            )

            if batch.anchor_ind.numel() > 0 and tri_weight > 0.0:
                anchor_arr = z[batch.anchor_ind,]
                positive_arr = z[batch.positive_ind,]
                negative_arr = z[batch.negative_ind,]
                sample_weight = batch.triplet_weight if (use_confidence_triplet and hasattr(batch, 'triplet_weight')) else None
                tri_output = _weighted_triplet_margin_loss(
                    anchor_arr,
                    positive_arr,
                    negative_arr,
                    margin=margin,
                    sample_weight=sample_weight,
                )
            else:
                tri_output = torch.tensor(0.0, device=device)

            if use_chain_consistency and batch.chain_anchor_ind.numel() > 0 and chain_weight > 0.0:
                chain_anchor_arr = z[batch.chain_anchor_ind,]
                chain_positive_arr = z[batch.chain_positive_ind,]
                chain_sample_weight = batch.chain_weight if batch.chain_weight.numel() > 0 else None
                chain_output = _weighted_positive_consistency_loss(
                    chain_anchor_arr,
                    chain_positive_arr,
                    sample_weight=chain_sample_weight,
                )
            else:
                chain_output = torch.tensor(0.0, device=device)

            loss = (
                lam_re * recon_loss
                + lam_rc * rc_loss
                + lam_dec * dec_loss
                + tri_weight * tri_output
                + chain_weight * chain_output
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

    #
    model.eval()
    adata.obsm[key_added], _, _ = _extract_dual_embeddings_cpu()
    return adata


from typing import Union, Tuple, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.typing import OptPairTensor, Adj, Size, NoneType, OptTensor
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


class GATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the attention coefficients :math:`\alpha_{i,j}` are computed as

    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True,
                 prune_weight: float = 0.0, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        # if isinstance(in_channels, int):
        self.lin_src = Linear(in_channels, heads * out_channels, False,
                              weight_initializer='glorot')
        self.lin_dst = Linear(in_channels, heads * out_channels, False,
                              weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.prune_weight = prune_weight
        self.attentions = None
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        torch.nn.init.xavier_uniform_(self.att)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
        return_attention_weights=None,
        prune_edge_index=None,
        tied_attention=None,
        tied_weights=None,
        attention=True,
    ) -> Tensor:

        H, C = self.heads, self.out_channels
        weight_src_override = None
        weight_dst_override = None
        if tied_weights is not None:
            weight_src_override, weight_dst_override = tied_weights

        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            if self.lin_dst is not None:
                if weight_src_override is not None:
                    x_src = x_dst = F.linear(x, weight_src_override).view(-1, H, C)
                else:
                    x_src = x_dst = self.lin_src(x).view(-1, H, C)
            else:
                if weight_src_override is not None:
                    x_src = F.linear(x, weight_src_override).view(-1, H, C)
                else:
                    x_src = self.lin_src(x).view(-1, H, C)
                x_dst = x_src if x[1] is None else self.lin_dst(x[1]).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            if weight_src_override is not None:
                x_src = F.linear(x_src, weight_src_override).view(-1, H, C)
            else:
                x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                if weight_dst_override is not None:
                    x_dst = F.linear(x_dst, weight_dst_override).view(-1, H, C)
                else:
                    x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        alpha_src = (x_src * self.att).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att).sum(dim=-1)
        alpha = (alpha_src, alpha_dst)
        self.attentions = alpha

        if tied_attention is not None:
            alpha = tied_attention

        if attention:
            assert alpha is not None
            if self.add_self_loops:
                if isinstance(edge_index, Tensor):
                    num_nodes = x_src.size(0)
                    if x_dst is not None:
                        num_nodes = min(num_nodes, x_dst.size(0))
                    num_nodes = min(size) if size is not None else num_nodes
                    edge_index, _ = remove_self_loops(edge_index)
                    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
                elif isinstance(edge_index, SparseTensor):
                    edge_index = set_diag(edge_index)

            if self.prune_weight == 0:
                out = self.propagate(edge_index, x=x, alpha=alpha, size=size)
            else:
                out = (1-self.prune_weight)*self.propagate(edge_index, x=x, alpha=alpha, size=size)+\
                      self.prune_weight*self.propagate(prune_edge_index, x=x, alpha=alpha, size=size)

            alpha = self._alpha
            assert alpha is not None
            self._alpha = None
        else:
            out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = torch.sigmoid(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class DVCAlignModel(torch.nn.Module):
    def __init__(self, hidden_dims):
        super().__init__()

        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConv(
            in_dim, num_hidden, heads=1, concat=False,
            dropout=0, add_self_loops=False, bias=False
        )
        self.conv2 = GATConv(
            num_hidden, out_dim, heads=1, concat=False,
            dropout=0, add_self_loops=False, bias=False
        )
        self.conv3 = GATConv(
            out_dim, num_hidden, heads=1, concat=False,
            dropout=0, add_self_loops=False, bias=False
        )
        self.conv4 = GATConv(
            num_hidden, in_dim, heads=1, concat=False,
            dropout=0, add_self_loops=False, bias=False
        )

        # Light residual branch from raw input to embedding space.
        self.residual = nn.Linear(in_dim, out_dim)

        # Learn a spot-wise gate so the model can adaptively balance
        # spatial-view and auxiliary-view embeddings instead of using
        # a fixed global fusion weight.
        self.view_gate = nn.Linear(out_dim * 2, 1)
        nn.init.zeros_(self.view_gate.weight)
        initial_gate = 0.7
        initial_gate = min(max(initial_gate, 1e-4), 1.0 - 1e-4)
        nn.init.constant_(self.view_gate.bias, np.log(initial_gate / (1.0 - initial_gate)))

    def forward(self, features, edge_index):
        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index, attention=False)

        residual = self.residual(features)
        h2 = 0.7 * h2 + 0.3 * residual

        decoder_weight_1 = (
            self.conv2.lin_src.weight.t(),
            self.conv2.lin_dst.weight.t(),
        )
        decoder_weight_2 = (
            self.conv1.lin_src.weight.t(),
            self.conv1.lin_dst.weight.t(),
        )
        h3 = F.elu(
            self.conv3(
                h2,
                edge_index,
                attention=True,
                tied_attention=self.conv1.attentions,
                tied_weights=decoder_weight_1,
            )
        )
        h4 = self.conv4(
            h3,
            edge_index,
            attention=False,
            tied_weights=decoder_weight_2,
        )

        return h2, h4

    def fuse_views(self, z_spatial, z_aux, fallback_weight=0.7, return_gate=False):
        if z_spatial.shape != z_aux.shape:
            raise ValueError(
                f"View embeddings must share the same shape, got {z_spatial.shape} and {z_aux.shape}."
            )

        gate_input = torch.cat([z_spatial, z_aux], dim=-1)
        alpha = torch.sigmoid(self.view_gate(gate_input))
        fused = alpha * z_spatial + (1.0 - alpha) * z_aux

        if return_gate:
            return fused, alpha
        return fused

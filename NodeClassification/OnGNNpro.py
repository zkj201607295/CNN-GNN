import torch
import torch.nn.functional as F
from mp_deterministic import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_sparse import SparseTensor, fill_diag

class ONGNNConv(MessagePassing):
    def __init__(self, tm_net, tm_norm, args):
        super(ONGNNConv, self).__init__('mean')
        self.args = args
        self.tm_net = tm_net
        self.tm_norm = tm_norm

    def forward(self, x, edge_index, last_tm_signal):
        if isinstance(edge_index, SparseTensor):
            edge_index = fill_diag(edge_index, fill_value=0)
            edge_index = fill_diag(edge_index, fill_value=1)
        else:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        m = self.propagate(edge_index, x=x)
        tm_signal_raw = F.softmax(self.tm_net(torch.cat((x, m), dim=1)), dim=-1)
        tm_signal_raw = torch.cumsum(tm_signal_raw, dim=-1)
        m_signal_raw = last_tm_signal+(1-last_tm_signal)*tm_signal_raw
        tm_signal = tm_signal_raw.repeat_interleave(repeats=int(self.args.hidden/96), dim=1)
        out = x*tm_signal + m*(1-tm_signal)

        out = self.tm_norm(out)

        return out, tm_signal_raw
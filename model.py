class GNNEncoder(nn.Module):
    def __init__(self, atom_dim, edge_dim, hidden_dim=256, num_layers=3):
        super().__init__()
        self.atom_emb = nn.Linear(atom_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, data):
        x = self.atom_emb(data.x)
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h_new = F.relu(conv(h, data.edge_index, data.edge_attr))
            h_new = bn(h_new)
            h = h + h_new
        h_out = torch.cat([global_mean_pool(h, data.batch),
                           global_max_pool(h, data.batch)], dim=1)
        return h_out

class PairInteractionBlock(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_in * 4, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
        )

    def forward(self, d, a):
        prod = d * a
        diff = torch.abs(d - a)
        x = torch.cat([d, a, prod, diff], dim=1)
        return self.mlp(x)

class PCEPredictor(nn.Module):
    def __init__(self, hidden_dim, phys_dim, global_dim, num_targets, atom_dim, edge_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_targets = num_targets

        self.gnn = GNNEncoder(atom_dim, edge_dim, hidden_dim=hidden_dim, num_layers=3)
        gnn_out_dim = hidden_dim * 2

        pair_dim = 256
        self.pair_dim = pair_dim

        self.proj_donor = nn.Sequential(
            nn.Linear(gnn_out_dim + global_dim, pair_dim),
            nn.ReLU(),
            nn.Linear(pair_dim, pair_dim),
            nn.ReLU(),
        )
        self.proj_acceptor = nn.Sequential(
            nn.Linear(gnn_out_dim + global_dim, pair_dim),
            nn.ReLU(),
            nn.Linear(pair_dim, pair_dim),
            nn.ReLU(),
        )

        self.pair_block = PairInteractionBlock(pair_dim, pair_dim)

        ff_in = pair_dim + phys_dim
        self.ff = nn.Sequential(
            nn.Linear(ff_in, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.head = nn.Linear(256, num_targets)
        self.log_vars = nn.Parameter(torch.zeros(num_targets))

    def forward(self, donor, acceptor, d_glob, a_glob, phys):
        h_d = self.gnn(donor)
        h_a = self.gnn(acceptor)

        d_cat = torch.cat([h_d, d_glob], dim=1)
        a_cat = torch.cat([h_a, a_glob], dim=1)

        d_repr = self.proj_donor(d_cat)
        a_repr = self.proj_acceptor(a_cat)

        pair_feat = self.pair_block(d_repr, a_repr)
        combined = torch.cat([pair_feat, phys], dim=1)

        h = self.ff(combined)
        out = self.head(h)

        pce_idx = self.num_targets - 1
        pce_col = out[:, pce_idx:pce_idx+1]
        pce_nonneg = F.softplus(pce_col)
        out = torch.cat([out[:, :pce_idx], pce_nonneg, out[:, pce_idx+1:]], dim=1)
        return out

    def multi_task_loss(self, preds, targets, pce_index=3, pce_boost=0.5, high_range_factor=1.0, high_thresh=12.5):
        mse_all = ((preds - targets) ** 2).mean(dim=0)
        precision = torch.exp(-self.log_vars)
        kendall = torch.sum(0.5 * precision * mse_all + 0.5 * self.log_vars)

        pce_mse = mse_all[pce_index]
        extra_pce = pce_boost * pce_mse

        mask_high = targets[:, pce_index] > high_thresh
        if mask_high.any():
            high_mse = ((preds[mask_high, pce_index] - targets[mask_high, pce_index])**2).mean()
            extra_high = high_range_factor * high_mse
        else:
            extra_high = 0.0

        total = kendall + extra_pce + extra_high
        return total
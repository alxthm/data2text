import math

import torch
import torch.utils.data
from dgl.nn.pytorch import edge_softmax
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from src.data.shared import Vocab, pad, NODE_TYPE


def replace_ent(x, ent, V, emb):
    device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
    # replace the entity
    mask = (x >= V).float()
    _x = emb((x * (1.0 - mask) + 3 * mask).long())  # 3 is <UNK>
    if mask.sum() == 0:
        return _x
    idx = ((x - V) * mask + 0 * (1.0 - mask)).long()
    return _x * (1.0 - mask[:, None]) + mask[:, None] * ent[
        torch.arange(len(idx)).to(device), idx
    ].view(_x.shape)


def len2mask(lens, device):
    max_len = max(lens)
    mask = torch.arange(max_len, device=device).unsqueeze(0).expand(len(lens), max_len)
    mask = mask >= torch.LongTensor(lens).to(mask).unsqueeze(1)
    return mask


class MSA(nn.Module):
    # Multi-head Self Attention
    def __init__(
        self,
        dim_h: int,
        dec_dim_in: int,
        n_head: int = None,
        head_dim: int = None,
        mode="normal",
    ):
        super(MSA, self).__init__()
        if mode == "copy":
            n_head, head_dim = 1, dim_h
            qninp, kninp = dec_dim_in, dim_h
        elif mode == "normal":
            qninp, kninp = dim_h, dim_h
        else:
            raise ValueError

        self.attn_drop = nn.Dropout(0.1)
        self.WQ = nn.Linear(
            qninp, n_head * head_dim, bias=True if mode == "copy" else False
        )
        if mode != "copy":
            self.WK = nn.Linear(kninp, n_head * head_dim, bias=False)
            self.WV = nn.Linear(kninp, n_head * head_dim, bias=False)
        self.n_head = n_head
        self.head_dim = head_dim
        self.mode = mode

    def forward(self, inp1, inp2, mask=None):
        B, L2, H = inp2.shape
        NH, HD = self.n_head, self.head_dim
        if self.mode == "copy":
            q, k, v = self.WQ(inp1), inp2, inp2
        else:
            q, k, v = self.WQ(inp1), self.WK(inp2), self.WV(inp2)
        # for ent_attn, inp1=_h with shape (bs, d)
        # for copy_attn, inp1=outs with shape (bs, max_sent_len, 2*d)
        L1 = 1 if inp1.ndim == 2 else inp1.shape[1]
        if self.mode != "copy":
            q = q / math.sqrt(H)  # why not in copy mode?
        q = q.view(B, L1, NH, HD).permute(0, 2, 1, 3)  # (B, NH, L1, HD)
        k = k.view(B, L2, NH, HD).permute(0, 2, 3, 1)  # (B, NH, HD, L2)
        v = v.view(B, L2, NH, HD).permute(0, 2, 1, 3)  # (B, NH, L2, HD)
        pre_attn = torch.matmul(q, k)  # (B, NH, L1, L2)
        if mask is not None:
            pre_attn = pre_attn.masked_fill(mask[:, None, None, :], -1e8)
        if self.mode == "copy":
            return pre_attn.squeeze(1)  # (B, L1, L2) since NH=1
        else:
            alpha = self.attn_drop(torch.softmax(pre_attn, -1))
            attn = (
                torch.matmul(alpha, v)  # (B, NH, L1, HD)
                .permute(0, 2, 1, 3)
                .contiguous()
                .view(B, L1, NH * HD)
            )
            ret = attn
            if inp1.ndim == 2:
                return ret.squeeze(1)
            else:
                return ret


class BiLSTM(nn.Module):
    # for entity encoding
    def __init__(self, dim_h: int, enc_lstm_layers: int, emb_drop: float):
        super(BiLSTM, self).__init__()
        self.drop = nn.Dropout(emb_drop)
        self.bilstm = nn.LSTM(
            dim_h,
            dim_h // 2,
            bidirectional=True,
            num_layers=enc_lstm_layers,
            batch_first=True,
        )

    def forward(self, inp, mask, ent_len=None):
        inp = self.drop(inp)  # (tot_ent, max_ent_len, dim_emb)
        lens = (mask == 0).sum(-1).long().tolist()
        pad_seq = pack_padded_sequence(
            inp, lens, batch_first=True, enforce_sorted=False
        )  # data of shape (sum(ent_len_ij), dim_emb)
        # y: packed sequence (dim=dim_h*2), h_t of the last layer for each t
        y, (_h, _c) = self.bilstm(pad_seq)
        # (tot_ent, num_layers*num_directions, dim_h), hidden state for t=seq_len
        _h = _h.transpose(0, 1).contiguous()
        # two directions of the top-layer  (tot_ent, dim_h*2=d)
        _h = _h[:, -2:].view(_h.size(0), -1)
        # _h.split: list of len bs, each element is a (num_ent_i, d) tensor
        ret = pad(_h.split(ent_len), out_type="tensor")  # (bs, max_num_ent, d)
        return ret


class GAT(nn.Module):
    # a graph attention network with dot-product attention
    def __init__(
        self, in_feats, out_feats, num_heads, ffn_drop=0.0, attn_drop=0.0, trans=True
    ):
        super(GAT, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.q_proj = nn.Linear(in_feats, num_heads * out_feats, bias=False)
        self.k_proj = nn.Linear(in_feats, num_heads * out_feats, bias=False)
        self.v_proj = nn.Linear(in_feats, num_heads * out_feats, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.ln1 = nn.LayerNorm(in_feats)
        self.ln2 = nn.LayerNorm(in_feats)
        if trans:
            self.FFN = nn.Sequential(
                nn.Linear(in_feats, 4 * in_feats),
                nn.PReLU(4 * in_feats),
                nn.Linear(4 * in_feats, in_feats),
                nn.Dropout(ffn_drop),
            )
            # I know it's a strange FFN
        self._trans = trans

    def forward(self, graph, feat):
        graph = graph.local_var()
        feat_c = feat.clone().detach().requires_grad_(False)
        q, k, v = self.q_proj(feat), self.k_proj(feat_c), self.v_proj(feat_c)
        q = q.view(-1, self._num_heads, self._out_feats)
        k = k.view(-1, self._num_heads, self._out_feats)
        v = v.view(-1, self._num_heads, self._out_feats)
        graph.ndata.update(
            {"ft": v, "el": k, "er": q}
        )  # k,q instead of q,k, the edge_softmax is applied on incoming edges
        # compute edge attention
        graph.apply_edges(fn_u_dot_v("el", "er", "e"))
        e = graph.edata.pop("e") / math.sqrt(self._out_feats * self._num_heads)
        graph.edata["a"] = edge_softmax(graph, e).unsqueeze(-1)
        # message passing
        graph.update_all(fn_u_mul_e("ft", "a", "m"), fn_sum("m", "ft2"))
        rst = graph.ndata["ft2"]
        # residual
        rst = rst.view(feat.shape) + feat
        if self._trans:
            rst = self.ln1(rst)
            rst = self.ln1(rst + self.FFN(rst))
            # use the same layer norm
        return rst


class GraphTrans(nn.Module):
    def __init__(self, dim_h: int, attn_drop: float, drop: float, prop: int):
        super().__init__()
        self.gat = nn.ModuleList(
            [
                GAT(
                    in_feats=dim_h,
                    out_feats=dim_h // 4,
                    num_heads=4,
                    attn_drop=attn_drop,
                    ffn_drop=drop,
                    trans=True,
                )
                for _ in range(prop)
            ]
        )
        self.prop = prop

    def forward(self, ent, ent_mask, ent_len, rel, rel_mask, graphs):
        device = ent.device
        ent_mask = ent_mask == 0  # reverse mask
        rel_mask = rel_mask == 0
        init_h = []
        for i in range(graphs.batch_size):
            init_h.append(ent[i][ent_mask[i]])
            init_h.append(rel[i][rel_mask[i]])
        init_h = torch.cat(init_h, 0)
        feats = init_h
        if graphs.number_of_nodes() != len(init_h):
            print("Err", graphs.number_of_nodes(), len(init_h), ent_mask, rel_mask)
        else:
            for i in range(self.prop):
                feats = self.gat[i](graphs, feats)
        g_root = feats.index_select(
            0,
            graphs.filter_nodes(lambda x: x.data["type"] == NODE_TYPE["root"]).to(
                device
            ),
        )
        g_ent = pad(
            feats.index_select(
                0,
                graphs.filter_nodes(
                    lambda x: x.data["type"] == NODE_TYPE["entity"]
                ).to(device),
            ).split(ent_len),
            out_type="tensor",
        )
        return g_ent, g_root


def fn_u_dot_v(n1, n2, n3):
    def func(edge_batch):
        return {
            n3: torch.matmul(
                edge_batch.src[n1].unsqueeze(-2), edge_batch.dst[n2].unsqueeze(-1)
            )
            .squeeze(-1)
            .squeeze(-1)
        }

    return func


def fn_u_mul_e(n1, n2, n3):
    def func(edge_batch):
        return {n3: edge_batch.src[n1] * edge_batch.data[n2]}

    return func


def fn_sum(n1, n2):
    def func(node_batch):
        return {n2: node_batch.mailbox[n1].sum(1)}

    return func


class G2T(nn.Module):
    def __init__(
        self,
        text_vocab: Vocab,
        ent_vocab: Vocab,
        rel_vocab: Vocab,
        dim_h: int,
        dim_z: int,
        enc_lstm_layers: int,
        n_head: int,
        head_dim: int,
        emb_dropout: float,
        attn_drop: float,
        drop: float,
        n_layers_gat: int,
        beam_max_len: int,
        length_penalty: float,
    ):
        super(G2T, self).__init__()
        self.text_vocab = text_vocab
        self.ent_vocab = ent_vocab
        self.rel_vocab = rel_vocab
        self.beam_max_len = beam_max_len
        self.length_penalty = length_penalty

        dec_dim_in = dim_h * 2
        self.dim_z = dim_z

        self.ent_emb = nn.Embedding(len(self.ent_vocab), dim_h, padding_idx=0)
        self.tar_emb = nn.Embedding(len(self.text_vocab), dim_h, padding_idx=0)
        nn.init.xavier_normal_(self.ent_emb.weight)
        self.rel_emb = nn.Embedding(len(self.rel_vocab), dim_h, padding_idx=0)
        nn.init.xavier_normal_(self.rel_emb.weight)

        # LSTM module takes an inout sequence and has an optimized for loop, while LSTMCell takes a single element
        # -> useful for decoder (https://stackoverflow.com/questions/57048120/pytorch-lstm-vs-lstmcell)
        self.decode_lstm = nn.LSTMCell(dec_dim_in + dim_z, dim_h)
        self.ent_enc = BiLSTM(dim_h, enc_lstm_layers, emb_dropout)
        self.graph_enc = GraphTrans(dim_h, attn_drop, drop, n_layers_gat)
        self.ent_attn = MSA(dim_h, dec_dim_in, n_head, head_dim)
        self.copy_attn = MSA(dim_h, dec_dim_in, mode="copy")
        self.blind = False
        self.copy_fc = nn.Linear(dec_dim_in, 1)
        self.pred_v_fc = nn.Linear(dec_dim_in, len(self.text_vocab))
        self.ln = nn.LayerNorm(dim_h)

        if dim_z > 0:
            # for q(z|x) ---> typo? actually used for p(z)
            self.vae_fc = nn.Linear(dim_h, dim_z * 2)
            # for p(z) ---> typo? actually used with vae_lstm for q(z|x)
            self.vae_pfc = nn.Linear(dim_h * 2, dim_z * 2)
            self.vae_lstm = nn.LSTM(dim_h, dim_h, batch_first=True, bidirectional=True)

    def enc_forward(self, batch_g2t, ent_mask, ent_text_mask, rel_mask):
        ent_enc = self.ent_enc(
            self.ent_emb(batch_g2t["ent_text"]), ent_text_mask, ent_len=batch_g2t["ent_len"]
        )  # (bs, max_num_ent, d)
        rel_emb = self.rel_emb(batch_g2t["rel"])  # (bs, max_num_rel, d)
        if self.blind:
            g_ent, g_root = ent_enc, ent_enc.mean(1)
        else:
            g_ent, g_root = self.graph_enc(
                ent_enc, ent_mask, batch_g2t["ent_len"], rel_emb, rel_mask, batch_g2t["graph"]
            )  # (bs, max_num_ent, d) and (bs, d)
        return self.ln(g_ent), g_root, ent_enc

    def get_vae_pz(self, inp):
        # compute p(z), note that the p(z) is learnable
        _z = self.vae_fc(inp)
        return (
            _z[:, : self.dim_z],
            _z[:, self.dim_z :],
        )  # mu and log_sigma

    def get_vae_qz(self, inp):
        # compute q(z|x)
        _z, _ = self.vae_lstm(inp)
        _z = self.vae_pfc(_z.mean(1))
        return _z[:, : self.dim_z], _z[:, self.dim_z :]

    def forward(self, batch_g2t, beam_size=-1):
        """

        Args:
            batch:
            beam_size: three modes
                - beam_size==-1 means training,
                - beam_size==1 means greedy decoding,
                - beam_size>1 means beam search

        Returns:
            During training (beam_size==-1), return (pred, torch.exp(pred_c), kl_div)
            with
                - pred of shape (bs, max_sent_len, len(text_vocab)+max_num_ent),
                the log probabilities of tokens at each time step
                (given previous target tokens)
                - pred_c of shape (bs, max_sent_len, max_num_ent), not used??
                - kl_div (scalar) = D_KL[q(z|x) || p(z|y)]

            During inference, simply return seq of shape (bs, max_sent_len) ?
            the predicted token indices using greedy/beam search decoding

        """

        # (bs, max_num_ent) bool tensor, with max_num_ent/rel the max nb of ent/rel in the batch sentences
        # False if entity j exists in sentence i (i.e. if j < num_ent_i)
        ent_mask = len2mask(batch_g2t["ent_len"], batch_g2t["ent_text"].device)
        ent_text_mask = batch_g2t["ent_text"] == 0  # (sum(num_ent_i), max_ent_len)

        rel_mask = batch_g2t["rel"] == 0  # (bs, max_num_rel), 0 means the <PAD>
        g_ent, g_root, ent_enc = self.enc_forward(
            batch_g2t, ent_mask, ent_text_mask, rel_mask
        )  # (bs, max_num_ent, d), except for g_root which is missing the 1st dim

        _h, _c = g_root, g_root.clone().detach()
        ctx = _h + self.ent_attn(_h, g_ent, mask=ent_mask)
        pmu, plog_sigma = self.get_vae_pz(g_root)

        if beam_size < 1:
            # training
            device = (
                torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
            )
            outs = []
            _mask = (
                batch_g2t["text"] >= len(self.text_vocab)
            ).long()  # 0 if token is in vocab, 1 if entity or unknown
            _inp = (
                _mask * 3 + (1.0 - _mask) * batch_g2t["text"]
            )  # 3 is <UNK>, otherwise use token index
            tar_inp = self.tar_emb(_inp.long())
            # Note: x[:,:,None] <-> unsqueeze(-1)

            # embeddings for tokens in text vocab (0. if unknown or entity)
            embeddings_text = (
                1.0 - _mask[:, :, None]
            ) * tar_inp  # (bs, max_sent_len, d)
            # embeddings for entity tokens (0. elsewhere)
            embeddings_ent = ent_enc[
                torch.arange(len(batch_g2t["text"]))[:, None].to(device),
                (
                    (batch_g2t["text"] - len(self.text_vocab)) * _mask
                ).long(),  # 0 for ENT_0 and other tokens, i for ENT_i
            ]  # (bs, max_sent_len, d)
            embeddings_ent = (
                embeddings_ent * _mask[:, :, None]
            )  # set to 0. if not entity
            tar_inp = embeddings_text + embeddings_ent
            mu, log_sigma = self.get_vae_qz(tar_inp)
            vae_z = torch.exp(0.5 * log_sigma) * torch.randn_like(log_sigma) + mu

            kld_loss = (
                (
                    0.5
                    * (
                        log_sigma
                        - plog_sigma
                        - 1
                        + torch.exp(plog_sigma) / (1e-6 + torch.exp(log_sigma))
                        + (mu - pmu) ** 2 / torch.exp(log_sigma)
                    )
                )
                .sum(-1)
                .mean()
            )

            tar_inp = tar_inp.transpose(0, 1)
            for t, xt in enumerate(tar_inp):
                _xt = torch.cat([ctx, xt, vae_z], 1)  # (bs, 2*d+dim_z)
                _h, _c = self.decode_lstm(_xt, (_h, _c))  # (bs, d)
                ctx = _h + self.ent_attn(_h, g_ent, mask=ent_mask)
                outs.append(torch.cat([_h, ctx], 1))
            outs = torch.stack(outs, 1)  # (bs, max_sent_len, 2*d)
            copy_gate = torch.sigmoid(self.copy_fc(outs))
            EPSI = 1e-6
            # copy
            pred_v = torch.log(copy_gate + EPSI) + torch.log_softmax(
                self.pred_v_fc(outs), -1
            )  # (bs, max_sent_len, len(text_vocab))
            pred_c = torch.log((1.0 - copy_gate) + EPSI) + torch.log_softmax(
                self.copy_attn(outs, ent_enc, mask=ent_mask), -1
            )  # (bs, max_sent_len, max_num_ent)
            pred = torch.cat([pred_v, pred_c], -1)
            return pred, torch.exp(pred_c), kld_loss
        else:
            if beam_size == 1:
                # greedy
                vae_z = torch.exp(0.5 * plog_sigma) * torch.randn_like(plog_sigma) + pmu
                device = g_ent.device
                B = g_ent.shape[0]
                seq = (
                    torch.ones(
                        B,
                    )
                    .long()
                    .to(device)
                    * self.text_vocab("<BOS>")
                ).unsqueeze(1)
                for t in range(self.beam_max_len):
                    _inp = seq[:, -1]
                    xt = replace_ent(
                        seq[:, -1], ent_enc, len(self.text_vocab), self.tar_emb
                    )
                    _xt = torch.cat([ctx, xt, vae_z], 1)
                    _h, _c = self.decode_lstm(_xt, (_h, _c))
                    ctx = _h + self.ent_attn(_h, g_ent, mask=ent_mask)
                    _y = torch.cat([_h, ctx], 1)
                    copy_gate = torch.sigmoid(self.copy_fc(_y))
                    pred_v = torch.log(copy_gate) + torch.log_softmax(
                        self.pred_v_fc(_y), -1
                    )
                    pred_c = torch.log((1.0 - copy_gate)) + torch.log_softmax(
                        self.copy_attn(_y.unsqueeze(1), ent_enc, mask=ent_mask).squeeze(
                            1
                        ),
                        -1,
                    )
                    pred = torch.cat([pred_v, pred_c], -1).view(B, -1)
                    for ban_item in ["<BOS>", "<PAD>", "<UNK>"]:
                        pred[:, self.text_vocab(ban_item)] = -1e8
                    _, word = pred.max(-1)
                    seq = torch.cat([seq, word.unsqueeze(1)], 1)
                    eos_idx = self.text_vocab("<EOS>")
                    if ((seq == eos_idx).float().max(-1)[0] == 1).all():
                        break
                return seq
            else:
                # beam search
                vae_z = torch.exp(0.5 * plog_sigma) * torch.randn_like(plog_sigma) + pmu
                device = g_ent.device
                B = g_ent.shape[0]
                BSZ = B * beam_size
                _h = _h.view(B, 1, -1).repeat(1, beam_size, 1).view(BSZ, -1)
                _c = _c.view(B, 1, -1).repeat(1, beam_size, 1).view(BSZ, -1)
                ent_mask = ent_mask.view(B, 1, -1).repeat(1, beam_size, 1).view(BSZ, -1)
                vae_z = vae_z.view(B, 1, -1).repeat(1, beam_size, 1).view(BSZ, -1)
                ctx = ctx.view(B, 1, -1).repeat(1, beam_size, 1).view(BSZ, -1)
                g_ent = (
                    g_ent.view(B, 1, g_ent.size(1), -1)
                    .repeat(1, beam_size, 1, 1)
                    .view(BSZ, g_ent.size(1), -1)
                )
                ent_enc = (
                    ent_enc.view(B, 1, ent_enc.size(1), -1)
                    .repeat(1, beam_size, 1, 1)
                    .view(BSZ, ent_enc.size(1), -1)
                )

                beam_best = torch.zeros(B).to(device) - 1e9
                beam_seq = (
                    torch.ones(B, beam_size).long().to(device)
                    * self.text_vocab("<BOS>")
                ).unsqueeze(-1)
                beam_best_seq = torch.zeros(B, 1).long().to(device)
                beam_score = torch.zeros(B, beam_size).to(device)
                done_flag = torch.zeros(B, beam_size).to(device)
                for t in range(self.beam_max_len):
                    _inp = beam_seq[:, :, -1].view(-1)
                    _mask = (_inp >= len(self.text_vocab)).long()
                    xt = replace_ent(
                        beam_seq[:, :, -1].view(-1),
                        ent_enc,
                        len(self.text_vocab),
                        self.tar_emb,
                    )
                    _xt = torch.cat([ctx, xt, vae_z], 1)
                    _h, _c = self.decode_lstm(_xt, (_h, _c))
                    ctx = _h + self.ent_attn(_h, g_ent, mask=ent_mask)
                    _y = torch.cat([_h, ctx], 1)
                    copy_gate = torch.sigmoid(self.copy_fc(_y))
                    pred_v = torch.log(copy_gate) + torch.log_softmax(
                        self.pred_v_fc(_y), -1
                    )
                    pred_c = torch.log((1.0 - copy_gate)) + torch.log_softmax(
                        self.copy_attn(_y.unsqueeze(1), ent_enc, mask=ent_mask).squeeze(
                            1
                        ),
                        -1,
                    )
                    pred = torch.cat([pred_v, pred_c], -1).view(B, beam_size, -1)
                    for ban_item in ["<BOS>", "<PAD>", "<UNK>"]:
                        pred[:, :, self.text_vocab(ban_item)] = -1e8
                    if t == self.beam_max_len - 1:  # force ending
                        tt = pred[:, :, self.text_vocab("<EOS>")]
                        pred = pred * 0 - 1e8
                        pred[:, :, self.text_vocab("<EOS>")] = tt
                    cum_score = beam_score.view(B, beam_size, 1) + pred
                    score, word = cum_score.topk(
                        dim=-1, k=beam_size
                    )  # B, beam_size, beam_size
                    score, word = score.view(B, -1), word.view(B, -1)
                    eos_idx = self.text_vocab("<EOS>")
                    if beam_seq.size(2) == 1:
                        new_idx = torch.arange(beam_size).to(device)
                        new_idx = new_idx[None, :].repeat(B, 1)
                    else:
                        _, new_idx = score.topk(dim=-1, k=beam_size)
                    new_src, new_score, new_word, new_done = [], [], [], []
                    LP = beam_seq.size(2) ** self.length_penalty  # length penalty
                    prefix_idx = torch.arange(B).to(device)[:, None]
                    new_word = word[prefix_idx, new_idx]
                    new_score = score[prefix_idx, new_idx]
                    _mask = (new_word == eos_idx).float()
                    _best = _mask * (done_flag == 0).float() * new_score
                    _best = _best * (_best != 0) - 1e8 * (_best == 0)
                    new_src = new_idx // beam_size
                    _best, _best_idx = _best.max(1)
                    _best = _best / LP
                    _best_mask = (_best > beam_best).float()
                    beam_best = beam_best * (1.0 - _best_mask) + _best_mask * _best
                    beam_best_seq = beam_best_seq * (
                        1.0 - _best_mask[:, None]
                    ) + _best_mask[:, None] * beam_seq[
                        prefix_idx, new_src[prefix_idx, _best_idx[:, None]]
                    ].squeeze(
                        1
                    )
                    new_score = -1e8 * _mask + (1.0 - _mask) * new_score
                    new_done = 1 * _mask + (1.0 - _mask) * done_flag
                    beam_score = new_score
                    done_flag = new_done
                    beam_seq = beam_seq.view(B, beam_size, -1)[
                        torch.arange(B)[:, None].to(device), new_src
                    ]
                    beam_seq = torch.cat([beam_seq, new_word.unsqueeze(2)], 2)
                    beam_best_seq = torch.cat(
                        [beam_best_seq, torch.zeros(B, 1).to(device)], 1
                    )
                    _h = _h.view(B, beam_size, -1)[
                        torch.arange(B)[:, None].to(device), new_src
                    ].view(BSZ, -1)
                    _c = _c.view(B, beam_size, -1)[
                        torch.arange(B)[:, None].to(device), new_src
                    ].view(BSZ, -1)
                    ctx = ctx.view(B, beam_size, -1)[
                        torch.arange(B)[:, None].to(device), new_src
                    ].view(BSZ, -1)
                    if (done_flag == 1).all():
                        break

                return beam_best_seq.long()

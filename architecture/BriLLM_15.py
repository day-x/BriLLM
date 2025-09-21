import math
import time
import jieba
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import io
import re
import random
from collections import defaultdict

jieba.re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%<>\-]+)", re.U)


class BriLLMNodeBias(nn.Module):
    def __init__(
            self,
            d_node,
            l_dataset=None,
            max_seq_len=512,
            inheritance_path=None,  # 加载模型
            inheritance_training=False,  # 训练
            inheritance_init=True,  # 是否支持初始化
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        super().__init__()
        self.temp_best_checkpoint = None  # 初始化 保存临时最优
        self.device = device
        self.d_node = d_node
        self.max_seq_len = max_seq_len

        self.amp_dtype = None

        ckpt = None
        if inheritance_path:
            if os.path.isfile(inheritance_path):
                print(f"load_model paht: {inheritance_path}")
                ckpt = torch.load(inheritance_path, map_location="cpu")
                self.word_to_id = ckpt["word_to_id"]
                self.id_to_word = ckpt["id_to_word"]
                self.edge2id = ckpt["edge2id"]
                self.d_node = ckpt["d_node"]
                self.vocab_size = len(self.word_to_id)
                self._init_special_ids()
                if "max_seq_len" in ckpt:
                    self.max_seq_len = int(ckpt["max_seq_len"])

                if inheritance_training:
                    if not l_dataset:
                        raise ValueError("训练已开启,但是无数据集传入")
                    if isinstance(l_dataset, (list, tuple)):
                        self.dataset_id = self.process_data(l_dataset[0])
                    else:
                        self.dataset_id = l_dataset[0]
            else:
                if inheritance_init:
                    print(f"模型:{inheritance_path} 未找到,跳过加载 正常初始化")
                    self.word_to_id, self.id_to_word = self.participle(l_dataset[0])
                    self.dataset_id = self.process_data(l_dataset[0])
                    self.vocab_size = len(self.word_to_id)
                    self.edge2id = self.auto_create_edge(self.dataset_id, l_dataset[1])
                else:
                    raise ValueError(f"模型:{inheritance_path} 未找到")
        else:
            # 正常按给定数据构建词表/数据/边
            self.word_to_id, self.id_to_word = self.participle(l_dataset[0])
            self.dataset_id = self.process_data(l_dataset[0])
            self.vocab_size = len(self.word_to_id)
            self.edge2id = self.auto_create_edge(self.dataset_id, l_dataset[1])

        edge2id_size = len(self.edge2id)

        # 参数
        self.bias_table = nn.Parameter(torch.empty(self.vocab_size, self.d_node).uniform_(-0.5, 0.5))
        self.W = nn.Parameter(torch.empty(edge2id_size, self.d_node, self.d_node).uniform_(-0.5, 0.5))
        self.bias = nn.Parameter(torch.empty(edge2id_size, self.d_node).uniform_(-0.5, 0.5))
        self.W_shared = nn.Parameter(torch.empty(self.d_node, self.d_node).uniform_(-0.5, 0.5))
        self.bias_shared = nn.Parameter(torch.empty(self.d_node).uniform_(-0.5, 0.5))
        self.a = nn.Parameter(torch.ones(1, self.max_seq_len, 1))
        self.gate = nn.Parameter(torch.tensor(0.1))
        self.pe_scale = nn.Parameter(torch.tensor(0.5))

        # PE cache
        self.register_buffer("PE_cache", self.get_positional_encoding(self.max_seq_len, self.d_node)[0])

        # 邻接缓存（在 auto_create_edge 里只在 CPU 构建为 LongTensor，这里迁移到 device）
        self._build_adj_cache_from_edge2id()  # 构建 self.adj_true_ids / self.adj_eidx (CPU)

        # eid_table（保持兼容）
        self.eid_table = self.build_lastid_to_eid_table()

        # 4) 如果有 ckpt，加载权重 + 词表/边
        if ckpt is not None:
            missing, unexpected = self._safe_load_state_dict(ckpt["model_state_dict"])
            self.word_to_id = ckpt["word_to_id"]
            self.id_to_word = ckpt["id_to_word"]
            self.edge2id = ckpt["edge2id"]
            self.eid_table = self.build_lastid_to_eid_table()
            self._build_adj_cache_from_edge2id()  # 重新构建邻接缓存（CPU）
            if missing or unexpected:
                print(f"[load] missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")

        print(f"V: {self.vocab_size}")
        self.to(self.device)

        self._W_pool = None
        self._b_pool = None
        self._pool_version = -1

    def _safe_load_state_dict(self, state_dict):
        own = self.state_dict()
        ok = {}
        for k, v in state_dict.items():
            if k in own and own[k].shape == v.shape:
                ok[k] = v
        missing = [k for k in own.keys() if k not in ok]
        unexpected = [k for k in state_dict.keys() if
                      k not in own or own.get(k, torch.empty(0)).shape != state_dict[k].shape]
        self.load_state_dict(ok, strict=False)
        return missing, unexpected

    def _init_special_ids(self):
        self.BOS_id = self.word_to_id.get('<BOS>', 0)
        self.END_id = self.word_to_id.get('<END>', 1)
        self.EOS_id = self.word_to_id.get('<EOS>', 2)
        self.UNK_id = self.word_to_id.get('<UNK>', 3)

    def _refresh_pools(self):
        self._W_pool = torch.cat([self.W, self.W_shared.unsqueeze(0)], dim=0)
        self._b_pool = torch.cat([self.bias, self.bias_shared.unsqueeze(0)], dim=0)

    def _get_pools(self):
        self._refresh_pools()
        return self._W_pool, self._b_pool

    def enable_compile(self, mode="reduce-overhead"):
        try:
            self.forward = torch.compile(self.forward, mode=mode, fullgraph=False)
            print(f"[compile] enabled: mode={mode}")
        except Exception as e:
            print(f"[compile] skipped: {e}")

    # ---------- 预测 ----------
    def _top_k_filter_logits(self, logits: torch.Tensor, k: int):
        if (k is None) or (k <= 0) or (k >= logits.numel()):
            return logits
        v, ix = torch.topk(logits, k)
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(0, ix, v)
        return mask

    def _top_p_filter_logits(self, logits: torch.Tensor, p: float):
        if (p is None) or (p <= 0.0) or (p >= 1.0):
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        keep = cumprobs <= p
        if not torch.any(keep):
            keep[0] = True
        filtered = torch.full_like(sorted_logits, float('-inf'))
        filtered[keep] = sorted_logits[keep]
        restored = torch.full_like(logits, float('-inf'))
        restored.scatter_(0, sorted_indices, filtered)
        return restored

    def predict(
            self,
            text,
            temperature=1.0,
            sample=False,
            top_k=None,
            top_p=None,
            max_len=50,
            pti=False
    ):
        self.eval()
        input_text = text
        if pti:
            print(f"输入: {text}")

        ids = [self.BOS_id] + self.words_to_ids(input_text) + [self.EOS_id]

        notInEdge = 0
        Edge = []
        for i in range(max_len):
            out = self.forward(ids)
            logits = out['logits']

            if sample and (temperature != 1.0 or top_k or top_p):
                logits_f = self._top_p_filter_logits(logits, top_p)
                logits_f = self._top_k_filter_logits(logits_f, top_k)
                probs = F.softmax(logits_f, dim=-1)
                if not torch.isfinite(probs).any():
                    probs = out['probs']
                next_id = torch.multinomial(probs, 1).item()
            else:
                probs = out['probs']
                next_id = torch.argmax(probs).item()

            if (ids[-1], next_id) not in self.edge2id:
                notInEdge += 1
                Edge.append(self.ids_to_words([(ids[-1], next_id)]))
            ids.append(next_id)
            yield self.ids_to_words([next_id])[0], i
            if next_id == self.END_id:
                break

    # ---------- 词表/数据/边 ----------
    def build_lastid_to_eid_table(self):
        # 兼容保留（不在热路径使用）
        return {(u, v): eidx for (u, v), eidx in self.edge2id.items()}

    def get_positional_encoding(self, seq_len, d_model):
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        position_encoding = torch.zeros(seq_len, d_model)
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        return position_encoding.unsqueeze(0).to(self.device)

    def auto_create_edge(self, dataset, minimum=-1):
        print(f"[处理边...]")
        edges = {}
        for i in range(len(dataset)):
            jv = dataset[i][0] + dataset[i][1]
            for j in range(1, len(jv)):
                if j + 1 < len(jv):
                    e = (jv[j], jv[j + 1])
                    edges[e] = edges.get(e, 0) + 1
        edge2id = {}
        allFrequencies = {}
        print(f"总边数量: {len(edges)}")
        for (u, v), freq in zip(edges.keys(), edges.values()):
            if u not in allFrequencies:
                allFrequencies[u] = [{v: (len(edge2id), freq)}]
            else:
                allFrequencies[u].append({v: (len(edge2id), freq)})
            if freq > minimum:
                edge2id[(u, v)] = len(edge2id)
        self.edges = edges
        if minimum and minimum != -1:
            print(f"处理后的数量: {len(edge2id)}")
        self.edge_freq = allFrequencies
        return edge2id

    def _build_adj_cache_from_edge2id(self):
        """基于 self.edge2id 构建邻接缓存（CPU Float/Long Tensor），在 .to() 里搬到目标设备"""
        tmp_vs = defaultdict(list)
        tmp_eid = defaultdict(list)
        for (u, v), eidx in self.edge2id.items():
            tmp_vs[u].append(v)
            tmp_eid[u].append(eidx)
        self.adj_true_ids = {}
        self.adj_eidx = {}
        for u in tmp_vs:
            self.adj_true_ids[u] = torch.tensor(tmp_vs[u], dtype=torch.long)
            self.adj_eidx[u] = torch.tensor(tmp_eid[u], dtype=torch.long)

    def words_to_ids(self, words):
        ids: list[int] = []
        for word in jieba.lcut(words):
            if word in self.word_to_id:
                ids.append(self.word_to_id[word])
            else:
                ids.append(self.UNK_id)
        return ids

    def ids_to_words(self, ids):
        words = []
        for i in ids:
            if i in self.id_to_word and i != self.UNK_id:
                words.append(self.id_to_word[i])
            else:
                words.append('<UNK>')
        return words

    def participle(self, texts):
        word_to_id = {'<BOS>': 0, '<END>': 1, '<EOS>': 2, '<UNK>': 3}
        self.BOS_id, self.END_id, self.EOS_id, self.UNK_id = 0, 1, 2, 3

        id_to_word = {idx: tok for tok, idx in word_to_id.items()}

        print(f"[分词...]")
        for i in tqdm(range(len(texts))):
            for j in jieba.lcut(texts[i]['query']):
                if j not in word_to_id:
                    l = len(word_to_id)
                    word_to_id[j] = l
                    id_to_word[l] = j
            for j in jieba.lcut(texts[i]['answer']):
                if j not in word_to_id:
                    l = len(word_to_id)
                    word_to_id[j] = l
                    id_to_word[l] = j
        return word_to_id, id_to_word

    def process_data(self, texts, max_answer_len=128):
        dataset = []
        print("[处理数据...]", end='')
        for words in tqdm(range(len(texts))):
            q = self.words_to_ids(texts[words]['query'])
            q.insert(0, 0)  # <BOS>
            q.append(2)     # <EOS>

            a = self.words_to_ids(texts[words]['answer'])
            a.append(1)  # <END>

            dataset.append((q, a))
        num = 0
        for i in dataset:
            num += len(i[0]) + len(i[1])
        print(f"->[数据token数量: {num}]")
        return dataset

    # ---------- 权重访问 ----------
    def get_bias(self, id1, id2):
        eid = self.edge2id.get((id1, id2), None)
        if eid is None:
            return self.bias_shared
        return self.bias[eid]

    def get_w(self, id1, id2):
        eid = self.edge2id.get((id1, id2), None)
        if eid is None:
            return self.W_shared
        return self.W[eid]

    def get_edge_next(self, id1, k=None):
        if id1 in self.edge_freq:
            ids = []
            if k:
                for i in self.edge_freq[id1]:
                    if list(i.values())[1] < k:
                        continue
                    ids.append(i)
                return ids
            else:
                return self.edge_freq[id1]
        else:
            return []

    def get_edge_next_d_node(self, id1):
        W = []
        b = []
        idv = []
        if id1 in self.edge_freq:
            for i in self.edge_freq[id1]:
                W.append(self.W[list(i.values())[0][0]])
                b.append(self.bias[list(i.values())[0][0]])
                idv.append(list(i.keys())[0])
            return torch.stack(W), torch.stack(b), idv
        else:
            return torch.stack([self.W_shared]), torch.stack([self.bias_shared]), idv

    def _forward_one(self, token_ids: list | torch.Tensor):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if not token_ids:
            raise ValueError("token_ids 不能为空")

        L = len(token_ids)
        if L > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
            L = self.max_seq_len

        e = []
        PE = self.PE_cache[:L]
        h0 = self.bias_table[token_ids] + self.pe_scale * PE

        for i in range(L):
            tid = token_ids[i]
            if i > 0:
                previousId = token_ids[i - 1]
                prev_state = self.gate * e[i - 1] + (1 - self.gate) * h0[i - 1]
                base = F.layer_norm(prev_state, (self.d_node,))
                x = F.gelu(
                    self.get_w(previousId, tid) @ base +
                    self.get_bias(previousId, tid) +
                    PE[i]
                )
                e.append(prev_state + x)
            else:
                base0 = h0[i]
                x0 = F.gelu(base0)
                e.append(base0 + x0)

        e = torch.stack(e, dim=0).to(self.device)
        A = F.softmax(self.a[:, :L, :], dim=1)
        E = torch.sum(A * e.unsqueeze(0), dim=1)
        lastId = token_ids[-1]

        W_pool, b_pool = self._get_pools()
        dev = E.device
        dt = E.dtype

        if lastId in self.adj_true_ids:
            true_id = self.adj_true_ids[lastId]
            eid_true = self.adj_eidx[lastId]
            W_true = self.W[eid_true].to(dev, dtype=dt).contiguous()
            b_true = self.bias[eid_true].to(dev, dtype=dt).contiguous()
            W_all = torch.cat([W_true, self.W_shared.unsqueeze(0).to(dev, dtype=dt)], dim=0).contiguous()
            b_all = torch.cat([b_true, self.bias_shared.unsqueeze(0).to(dev, dtype=dt)], dim=0).contiguous()
            true_id = true_id.to(dev)
        else:
            W_all = self.W_shared.unsqueeze(0).to(dev, dtype=dt)
            b_all = self.bias_shared.unsqueeze(0).to(dev, dtype=dt)
            true_id = torch.empty(0, dtype=torch.long, device=dev)

        E_vec = E.squeeze(0).to(dev, dtype=dt).contiguous()
        PE_last = PE[L - 1].to(dev, dtype=dt)

        if not (torch.isfinite(W_all).all() and torch.isfinite(b_all).all() and torch.isfinite(E_vec).all()):
            raise RuntimeError("NaN/Inf detected in inputs to matmul")

        y = torch.matmul(W_all, E_vec) + b_all + PE_last
        y = F.gelu(y)

        y_all = y.new_empty(self.vocab_size, self.d_node)
        y_all[:] = y[-1]
        if true_id.numel() > 0:
            y_all[true_id] = y[:-1]

        v_predict = y_all.norm(p=2, dim=1)
        logits = v_predict - v_predict.max()
        return logits

    def get_eid_row(self, lastId):
        row = torch.full((self.vocab_size,), self.W.shape[0], dtype=torch.long)
        for (u, v), eidx in self.edge2id.items():
            if u == lastId:
                row[v] = eidx
        return row

    # ---------- Forward（批） ----------
    def _forward_batch(self, batch_token_ids):
        device = self.device
        pad_id = getattr(self, "pad_id", 0)

        if isinstance(batch_token_ids, torch.Tensor):
            assert batch_token_ids.dtype in (torch.long, torch.int64)
            seqs = batch_token_ids.tolist()
        else:
            seqs = batch_token_ids
        if not seqs or any(len(s) == 0 for s in seqs):
            raise ValueError("batch_token_ids 不能为空或包含空序列")

        seqs_trunc, lengths = [], []
        for s in seqs:
            if len(s) > self.max_seq_len:
                s = s[: self.max_seq_len]
            seqs_trunc.append(s)
            lengths.append(len(s))

        B = len(seqs_trunc)
        L_max = max(lengths)
        d = self.d_node

        token_tensor = torch.full((B, L_max), pad_id, dtype=torch.long, device=device)
        for i, s in enumerate(seqs_trunc):
            token_tensor[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)

        lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
        pad_mask = torch.arange(L_max, device=device).unsqueeze(0) < lengths_t.unsqueeze(1)

        PE = self.PE_cache[:L_max].to(device)
        h0 = self.bias_table[token_tensor] + self.pe_scale * PE.unsqueeze(0)

        base0 = h0[:, 0, :]
        e0 = base0 + F.gelu(base0)
        e_list = [e0]

        W_pool, b_pool = self._get_pools()

        for i in range(1, L_max):
            prev_state = self.gate * e_list[i - 1] + (1.0 - self.gate) * h0[:, i - 1, :]
            base = F.layer_norm(prev_state, (d,))

            prev_ids = token_tensor[:, i - 1]
            ids = token_tensor[:, i]

            eid_batch = [
                self.eid_table.get((int(u), int(v)), self.W.shape[0])
                for u, v in zip(prev_ids.tolist(), ids.tolist())
            ]
            eid_batch = torch.tensor(eid_batch, device=self.device, dtype=torch.long)

            W = torch.cat([self.W, self.W_shared.unsqueeze(0)], dim=0)[eid_batch]
            b = torch.cat([self.bias, self.bias_shared.unsqueeze(0)], dim=0)[eid_batch]

            Wx = torch.bmm(W, base.unsqueeze(-1)).squeeze(-1)
            x = F.gelu(Wx + b + PE[i].unsqueeze(0))

            valid_i = pad_mask[:, i].unsqueeze(-1)
            e_next = torch.where(valid_i, prev_state + x, e_list[i - 1])
            e_list.append(e_next)

        e = torch.stack(e_list, dim=1)

        base_A = self.a[:, :L_max, :].to(device).squeeze(0)
        A_logits = base_A.unsqueeze(0).expand(B, -1, -1)
        A_logits = A_logits.masked_fill(~pad_mask.unsqueeze(-1), float("-inf"))
        A = F.softmax(A_logits, dim=1)

        E = torch.sum(A * e, dim=1)

        last_indices = (lengths_t - 1).clamp(min=0)
        last_ids = token_tensor.gather(1, last_indices.view(B, 1)).squeeze(1)

        logits_out = []
        dev = E.device
        dt = E.dtype

        for b_idx in range(B):
            lastId = int(last_ids[b_idx].item())

            if lastId in self.adj_true_ids:
                true_id = self.adj_true_ids[lastId]
                eid_true = self.adj_eidx[lastId]

                W_true = self.W[eid_true].to(dev, dtype=dt).contiguous()
                b_true = self.bias[eid_true].to(dev, dtype=dt).contiguous()
                W_unk = self.W_shared.to(dev, dtype=dt).unsqueeze(0).contiguous()
                b_unk = self.bias_shared.to(dev, dtype=dt).unsqueeze(0).contiguous()

                W_all_small = torch.cat([W_true, W_unk], dim=0).contiguous()
                b_all_small = torch.cat([b_true, b_unk], dim=0).contiguous()

                true_id = true_id.to(dev)
            else:
                W_all_small = self.W_shared.to(dev, dtype=dt).unsqueeze(0).contiguous()
                b_all_small = self.bias_shared.to(dev, dtype=dt).unsqueeze(0).contiguous()
                true_id = torch.empty(0, dtype=torch.long, device=dev)

            E_vec = E[b_idx].to(dev, dtype=dt).contiguous()
            PE_last = PE[last_indices[b_idx]].to(dev, dtype=dt)

            y_small = torch.matmul(W_all_small, E_vec).add(b_all_small).add(PE_last)
            y_small = F.gelu(y_small)

            y_unk = y_small[-1]
            y_all = y_small.new_empty(self.vocab_size, self.d_node)
            y_all[:] = y_unk
            if true_id.numel() > 0:
                y_all[true_id] = y_small[:-1]

            v_predict = y_all.norm(p=2, dim=1)
            logits = v_predict - v_predict.max()
            logits_out.append(logits)

        return torch.stack(logits_out, dim=0)

    # ---------- forward 接口 ----------
    def forward(self, token_ids: list):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # 批
        if isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], (list, tuple)):
            logits = self._forward_batch(list(token_ids))
            probs = F.softmax(logits, dim=1)
            return {"logits": logits, "probs": probs, "loss": None}

        # 单样本
        logits = self._forward_one(list(token_ids))
        probs = F.softmax(logits, dim=-1)
        return {"logits": logits, "probs": probs, "loss": None}

    # ---------- 训练 ----------
    def train_model(self, epochs, dataset=None, save_model_path=None, batch=None, use_mixed_precision=False, lr=0.0015,
                    precision_dtype=None, max_auto_batch=-1, stop_loss=1e-3, use_temp_best=False, disrupt=False):
        """
        :param epochs: 训练轮次
        :param dataset: 数据集(处理后)
        :param save_model_path: 保存路径
        :param batch: 训练批次 固定/自动
        :param use_mixed_precision: 使用混合精度训练
        :param lr: 学习率
        :param precision_dtype: 混合精度 精度
        :param max_auto_batch: 自动批次下的最大批次
        :param stop_loss: 停止 loss
        :param use_temp_best: 变量保存最有模型
        :param disrupt: 打乱数据集
        """
        global batchDataset, batchY, scaler, best_loss
        if save_model_path:
            best_loss = float("inf")

        self.train()
        if not dataset:
            dataset = self.dataset_id

        try:
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, fused=True)
        except TypeError:
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        criterion = nn.CrossEntropyLoss()

        if use_mixed_precision:
            cc_major, cc_minor = torch.cuda.get_device_capability()
            if cc_major < 7 and use_mixed_precision != 'all':
                raise ValueError("当前算力架构不支持混合精度,如果您执意要使用 请设置 use_mixed_precision=all")
            elif not precision_dtype:
                if (cc_major, cc_minor) == (7, 0):
                    precision_dtype = torch.float16
                else:
                    precision_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.amp_dtype = precision_dtype
            scaler = torch.amp.GradScaler(device="cuda")

        isBatch = batch == "auto" or (batch is not None and batch > 1)

        if disrupt:
            random.shuffle(dataset)
            if batch == "auto":
                print("提示：我们不推荐您同时使用batch和disrupt. auto推荐您在小规模训练时使用")

        if isBatch:
            batchDataset = []
            batchY = []
            print(f"[处理batch数据]")
            for i in range(len(dataset)):
                q, a = dataset[i]
                prefixes = [q + a[:j] for j in range(len(a))]
                targets = a
                for j in range(len(prefixes)):
                    batchDataset.append(prefixes[j])
                    batchY.append(targets[j])
            if len(batchDataset) != len(batchY):
                raise ValueError(f"[错误] batchDataset({len(batchDataset)}) 与 batchY({len(batchY)}) 长度不一致")
        elif use_mixed_precision:
            raise ValueError("[错误] 当前模式(无batch)不支持混合精度")

        # 训练
        for epoch in range(epochs):
            startTime = time.time()
            lossItem = None

            if isBatch:
                def _for_batch(batch_x, batch_t):
                    assert len(batch_x) == len(batch_t)
                    n = len(batch_x)
                    start = 0
                    last_loss = None

                    def _try_run(x, t):
                        optimizer.zero_grad(set_to_none=True)
                        if use_mixed_precision:
                            with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                                out = self.forward(x)
                                logits = out["logits"]
                                targets = torch.tensor(t, dtype=torch.long, device=logits.device)
                                loss = criterion(logits, targets)
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            out = self.forward(x)
                            logits = out["logits"]
                            targets = torch.tensor(t, dtype=torch.long, device=logits.device)
                            loss = criterion(logits, targets)
                            loss.backward()
                            optimizer.step()
                        return float(loss.item())

                    while start < n:
                        end = n
                        while True:
                            try:
                                if end <= start:
                                    raise RuntimeError("Empty sub-batch during dynamic split")
                                sub_x = x = batch_x[start:end]
                                sub_t = t = batch_t[start:end]
                                last_loss = _try_run(sub_x, sub_t)
                                start = end
                                break
                            except RuntimeError as e:
                                msg = str(e).lower()
                                if ("out of memory" in msg or "cuda oom" in msg) and torch.cuda.is_available():
                                    optimizer.zero_grad(set_to_none=True)
                                    torch.cuda.empty_cache()
                                    new_end = start + max(1, (end - start) // 2)
                                    if new_end >= end:
                                        raise
                                    end = new_end
                                    continue
                                else:
                                    raise
                    return last_loss

                if batch == "auto":
                    for i in tqdm(range(len(dataset))):
                        q, a = dataset[i]
                        prefixes = [q + a[:j] for j in range(len(a))]
                        L = len(prefixes)
                        if (max_auto_batch is not None and max_auto_batch != -1) and len(prefixes) > max_auto_batch:
                            for s in range(0, len(prefixes), max_auto_batch):
                                batch_x = prefixes[s: s + max_auto_batch]
                                batch_t = a[s: s + max_auto_batch]
                                lossItem = _for_batch(batch_x, batch_t)
                        else:
                            offset = sum(len(dataset[k][1]) for k in range(i))
                            batch_x = batchDataset[offset: offset + L]
                            batch_t = batchY[offset: offset + L]
                            lossItem = _for_batch(batch_x, batch_t)
                else:
                    for i in tqdm(range(0, len(batchDataset), batch)):
                        batch_x = batchDataset[i:i + batch]
                        batch_t = batchY[i:i + batch]
                        lossItem = _for_batch(batch_x, batch_t)

            else:
                for i in tqdm(range(len(dataset))):
                    q, a = dataset[i]
                    for j in range(len(a)):
                        x = q + a[:j]
                        y = a[j]
                        out = self.forward(x)
                        logits = out["logits"].unsqueeze(0)
                        target = torch.tensor([y], dtype=torch.long, device=logits.device)
                        loss = criterion(logits, target)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    lossItem = loss.item()

            if use_temp_best:
                self.temp_best_checkpoint = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

            if save_model_path:
                if lossItem is not None and lossItem < best_loss:
                    best_loss = lossItem
                    self.save_model(
                        path=save_model_path,
                        optimizer=optimizer.state_dict(),
                        epoch=epoch,
                        loss=lossItem,
                    )
                    print(f"🔥 新最佳模型已保存 (loss={best_loss:.6f})")

            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {lossItem:.6f}, Time: {time.time() - startTime:.2f}s")

            if lossItem <= stop_loss:
                print(f"低于:{stop_loss},自动退出")
                break

        if save_model_path:
            print(f"自动保存模型Loss: {best_loss:.6f}")

    def save_model(self, path, optimizer=None, epoch=None, loss=None):
        ckpt = {
            "model_state_dict": self.state_dict(),
            "d_node": self.d_node,
            "vocab_size": self.vocab_size,
            "word_to_id": self.word_to_id,
            "id_to_word": self.id_to_word,
            "edge2id": self.edge2id,
            "max_seq_len": self.max_seq_len,
        }
        if optimizer:
            ckpt["optimizer_state_dict"] = optimizer if isinstance(optimizer, dict) else optimizer.state_dict()
        if epoch is not None:
            ckpt["epoch"] = epoch
        if loss is not None:
            ckpt["loss"] = loss
        torch.save(ckpt, path)

    def load_best_model(self):
        ckpt = getattr(self, "temp_best_checkpoint", None)
        if ckpt is None:
            print("[warn] 没有可用的临时最优权重，load_best_model() 被忽略。")
            return
        self.load_state_dict(ckpt)
        print("[info] 已加载临时最优模型。")

    @staticmethod
    def load_model(path_or_bytes, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        if isinstance(path_or_bytes, (str, os.PathLike)):
            ckpt = torch.load(path_or_bytes, map_location="cpu")
            path_for_inherit = str(path_or_bytes)
        elif isinstance(path_or_bytes, (bytes, bytearray)):
            ckpt = torch.load(io.BytesIO(path_or_bytes), map_location="cpu")
            path_for_inherit = None
        else:
            raise TypeError(
                f"load_model 需要文件路径(str/PathLike)或 bytes，收到: {type(path_or_bytes)}。"
            )

        model = BriLLMNodeBias(
            d_node=ckpt["d_node"],
            l_dataset=None,
            max_seq_len=ckpt.get("max_seq_len", 512),
            inheritance_path=path_for_inherit,
            inheritance_training=False,
            inheritance_init=True,
            device=device
        )

        if path_for_inherit is None:
            model._safe_load_state_dict(ckpt["model_state_dict"])
            model.word_to_id = ckpt["word_to_id"]
            model.id_to_word = ckpt["id_to_word"]
            model.edge2id = ckpt["edge2id"]
            model._init_special_ids()
            model._build_adj_cache_from_edge2id()
            model.eid_table = model.build_lastid_to_eid_table()

        return model, ckpt

    def to(self, device):
        super().to(device)
        self.device = device
        if hasattr(self, "adj_true_ids"):
            for k in list(self.adj_true_ids.keys()):
                self.adj_true_ids[k] = self.adj_true_ids[k].to(device)
                self.adj_eidx[k] = self.adj_eidx[k].to(device)
        self._refresh_pools()
        return self

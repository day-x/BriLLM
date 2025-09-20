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
        # self.vocab_size = vocab_size

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
                # 如果 ckpt 里保存了 max_seq_len，可覆盖（可选）
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
            # 2) 正常按给定数据构建词表/数据/边
            self.word_to_id, self.id_to_word = self.participle(l_dataset[0])
            self.dataset_id = self.process_data(l_dataset[0])
            self.vocab_size = len(self.word_to_id)
            self.edge2id = self.auto_create_edge(self.dataset_id, l_dataset[1])

        edge2id_size = len(self.edge2id)

        self.bias_table = nn.Parameter(torch.empty(self.vocab_size, self.d_node).uniform_(-0.5, 0.5))

        self.W = nn.Parameter(torch.empty(edge2id_size, self.d_node, self.d_node).uniform_(-0.5, 0.5))
        self.bias = nn.Parameter(torch.empty(edge2id_size, self.d_node).uniform_(-0.5, 0.5))

        self.W_shared = nn.Parameter(torch.empty(self.d_node, self.d_node).uniform_(-0.5, 0.5))
        self.bias_shared = nn.Parameter(torch.empty(self.d_node).uniform_(-0.5, 0.5))

        self.a = nn.Parameter(torch.ones(1, self.max_seq_len, 1))

        self.gate = nn.Parameter(torch.tensor(0.1))
        self.pe_scale = nn.Parameter(torch.tensor(0.5))

        self.register_buffer("PE_cache", self.get_positional_encoding(self.max_seq_len, self.d_node)[0])
        self.eid_table = self.build_lastid_to_eid_table()
        # self.register_buffer("eid_table", self.build_lastid_to_eid_table())

        # 4) 如果有 ckpt，加载权重 + 词表/边
        if ckpt is not None:
            missing, unexpected = self._safe_load_state_dict(ckpt["model_state_dict"])
            # 覆盖词典和边（再次保证一致）
            self.word_to_id = ckpt["word_to_id"]
            self.id_to_word = ckpt["id_to_word"]
            self.edge2id = ckpt["edge2id"]
            # 重建 eid_table（基于 ckpt 的 edge2id）
            self.eid_table = self.build_lastid_to_eid_table()
            # self.register_buffer("eid_table", self.build_lastid_to_eid_table())
            if missing or unexpected:
                print(f"[load] missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
        # self.edge2id_len = None
        print(f"V: {self.vocab_size}")
        # print(self.edge2id)
        self.to(self.device)

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
        # 如果词表里没有，就给出默认回退值，避免 KeyError
        self.BOS_id = self.word_to_id.get('<BOS>', 0)
        self.END_id = self.word_to_id.get('<END>', 1)
        self.EOS_id = self.word_to_id.get('<EOS>', 2)
        self.UNK_id = self.word_to_id.get('<UNK>', 3)

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
        # 排序（大->小）
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)

        # 保留累计概率 <= p 的 token（再确保至少保留一个）
        keep = cumprobs <= p
        if not torch.any(keep):
            keep[0] = True  # 至少留一个

        # 把不保留的 logits 设为 -inf
        filtered = torch.full_like(sorted_logits, float('-inf'))
        filtered[keep] = sorted_logits[keep]

        # 还原到原索引位置
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
            # if i == max_len - 1:
            # print(f"输出不存在的边数量: {notInEdge}")
            # print(f"边:{Edge}")
            if next_id == self.END_id:
                # print(f"输出不存在的边数量: {notInEdge}")
                # print(f"边:{Edge}")
                break

    def build_lastid_to_eid_table(self):
        # 不再生成 (V,V)，而是直接存一个 dict
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
        # 统计频率
        edges = {}
        for i in range(len(dataset)):
            # 从下标1开始 去头
            jv = dataset[i][0] + dataset[i][1]
            for j in range(1, len(jv)):
                if j + 1 < len(jv):
                    if (jv[j], jv[j + 1]) not in edges:
                        edges[(jv[j], jv[j + 1])] = 1
                    else:
                        edges[(jv[j], jv[j + 1])] += 1
        edge2id = {}
        allFrequencies = {}
        print(f"总边数量: {len(edges)}")
        for i in zip(edges.keys(), edges.values()):
            id1 = i[0][0]
            id2 = i[0][1]
            freq = i[1]
            if id1 not in allFrequencies:
                allFrequencies[id1] = [{id2: (len(edge2id), freq)}]
            elif id1 in allFrequencies:
                allFrequencies[id1].append({id2: (len(edge2id), freq)})
            if i[1] > minimum:
                edge2id[i[0]] = len(edge2id)
        self.edges = edges
        if minimum and minimum != -1: print(f"处理后的数量: {len(edge2id)}")
        # self.edge_len = len(edge2id)
        # print(allFrequencies)
        self.edge_freq = allFrequencies
        return edge2id

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
            q.insert(0, 0)  # <SOS>
            q.append(2)  # <EOS>

            # Answer + <END>
            a = self.words_to_ids(texts[words]['answer'])
            # a = a[:max_answer_len]  # 截断到最大长度
            a.append(1)  # <END>

            dataset.append((q, a))
        num = 0
        for i in dataset:
            for _ in i[0]:
                num += 1
            for _ in i[1]:
                num += 1
        print(f"->[数据token数量: {num}]")
        return dataset

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
        # {id: (位置, 频率)}
        if id1 in self.edge_freq:
            ids = []
            if k:
                for i in self.edge_freq[id1]:
                    # print(i)
                    # ids.extend(list(i.keys()))
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
        id = []
        if id1 in self.edge_freq:
            for i in self.edge_freq[id1]:
                # print(list(i.values())[0][0])
                W.append(self.W[list(i.values())[0][0]])
                b.append(self.bias[list(i.values())[0][0]])
                id.append(list(i.keys())[0])
            return torch.stack(W), torch.stack(b), id
        else:
            return torch.stack([self.W_shared]), torch.stack([self.bias_shared]), id

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
            id = token_ids[i]
            if i > 0:
                previousId = token_ids[i - 1]
                prev_state = self.gate * e[i - 1] + (1 - self.gate) * h0[i - 1]
                base = F.layer_norm(prev_state, (self.d_node,))
                x = F.gelu(
                    self.get_w(previousId, id) @ base +
                    self.get_bias(previousId, id) +
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

        Wb_W, Wb_b, true_id = self.get_edge_next_d_node(lastId)

        dev = E.device
        dt = E.dtype

        Wb_W = Wb_W.to(dev, dtype=dt).contiguous()
        Wb_b = Wb_b.to(dev, dtype=dt).contiguous()
        true_id = torch.as_tensor(true_id, device=dev, dtype=torch.long)

        W_shared = self.W_shared.to(dev, dtype=dt).unsqueeze(0).contiguous()
        b_shared = self.bias_shared.to(dev, dtype=dt).unsqueeze(0).contiguous()

        W_all = torch.cat([Wb_W, W_shared], dim=0).contiguous()
        b_all = torch.cat([Wb_b, b_shared], dim=0).contiguous()

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
        temperature = 1.0
        logits = v_predict / max(temperature, 1e-6)
        logits = logits - logits.max()
        return logits

    def get_eid_row(self, lastId):
        row = torch.full((self.vocab_size,), self.W.shape[0], dtype=torch.long)  # 默认 unknown
        for (u, v), eidx in self.edge2id.items():
            if u == lastId:
                row[v] = eidx
        return row

    def _forward_batch(self, batch_token_ids):
        global W_true, b_true
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

        # e_0
        base0 = h0[:, 0, :]
        e0 = base0 + F.gelu(base0)

        e_list = [e0]

        W_pool = torch.cat([self.W, self.W_shared.unsqueeze(0)], dim=0)
        b_pool = torch.cat([self.bias, self.bias_shared.unsqueeze(0)], dim=0)

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

            W = W_pool[eid_batch]
            b = b_pool[eid_batch]

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
        E_len = self.W.shape[0]
        dev = E.device
        dt = E.dtype

        for b_idx in range(B):
            lastId = int(last_ids[b_idx].item())
            eid_full = self.get_eid_row(lastId)
            eid_full = eid_full.to(dev)

            true_mask = (eid_full != E_len)
            true_id = true_mask.nonzero(as_tuple=False).squeeze(1)
            K = true_id.numel()

            if K > 0:
                eid_true = eid_full[true_id]
                W_true = self.W[eid_true].to(dev, dtype=dt).contiguous()
                b_true = self.bias[eid_true].to(dev, dtype=dt).contiguous()
            W_unk = self.W_shared.to(dev, dtype=dt).unsqueeze(0).contiguous()
            b_unk = self.bias_shared.to(dev, dtype=dt).unsqueeze(0).contiguous()

            if K > 0:
                W_all_small = torch.cat([W_true, W_unk], dim=0).contiguous()
                b_all_small = torch.cat([b_true, b_unk], dim=0).contiguous()
            else:
                W_all_small = W_unk
                b_all_small = b_unk

            E_vec = E[b_idx].to(dev, dtype=dt).contiguous()
            PE_last = PE[last_indices[b_idx]].to(dev, dtype=dt)

            y_small = torch.matmul(W_all_small, E_vec).add(b_all_small).add(PE_last)
            y_small = F.gelu(y_small)

            y_unk = y_small[-1]
            y_all = y_small.new_empty(self.vocab_size, self.d_node)
            y_all[:] = y_unk
            if K > 0:
                y_all[true_id] = y_small[:-1]

            v_predict = y_all.norm(p=2, dim=1)
            logits = v_predict - v_predict.max()
            logits_out.append(logits)

        return torch.stack(logits_out, dim=0)

    def forward(self, token_ids: list):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], (list, tuple)):
            logits = self._forward_batch(list(token_ids))
            probs = F.softmax(logits, dim=1)
            return {
                "logits": logits,
                "probs": probs,
                # "pred_id": pred,
                "loss": None
            }

        logits = self._forward_one(list(token_ids))
        probs = F.softmax(logits, dim=-1)
        # pred = torch.argmax(logits).item()
        return {
            "logits": logits,
            "probs": probs,
            # "pred_id": pred,
            "loss": None
        }

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
        :return:
        """
        global batchDataset, batchY, scaler, best_loss
        if save_model_path:
            best_loss = float("inf")  # 初始为正无穷

        self.train()
        if not dataset:
            # print(f"train_model->dataset:None->>[使用默认数据]")
            dataset = self.dataset_id

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        if use_mixed_precision:
            cc_major, cc_minor = torch.cuda.get_device_capability()
            if cc_major < 7 and use_mixed_precision != 'all':
                raise ValueError("当前算力架构不支持混合精度,如果您执意要使用 请设置 use_mixed_precision=all")
            elif not precision_dtype:
                if (cc_major, cc_minor) == (7, 0):
                    precision_dtype = torch.float16
                else:
                    if torch.cuda.is_bf16_supported():
                        precision_dtype = torch.bfloat16
                    else:
                        precision_dtype = torch.float16

            scaler = torch.amp.GradScaler(device="cuda")
        # print(precision_dtype)

        isBatch = batch == "auto" or (batch is not None and batch > 1)

        if disrupt:
            random.shuffle(dataset)
            if batch == "auto":
                print(f"提示："
                      f"我们不推荐您同时使用batch和disrupt."
                      f"auto推荐您在小规模训练时使用")

        if isBatch:
            batchDataset = []
            batchY = []
            print(f"[处理batch数据]")
            for i in range(len(dataset)):
                if isBatch:
                    q, a = dataset[i]
                    prefixes = [q + a[:j] for j in range(len(a))]
                    targets = torch.tensor(a, dtype=torch.long, device=self.device)
                    for j in range(len(prefixes)):
                        batchDataset.append(
                            prefixes[j]
                        )
                        batchY.append(
                            targets[j]
                        )
                if len(batchDataset) != len(batchY):
                    raise ValueError(
                        f"[错误] batchDataset({len(batchDataset)}) 与 batchY({len(batchY)}) 长度不一致，"
                        "请检查数据处理逻辑。"
                    )
        elif use_mixed_precision:
            raise ValueError(
                f"[错误] 当前模式(无batch)不支持混合精度"
            )
        for epoch in range(epochs):
            startTime = time.time()
            lossItem = None
            if isBatch:
                def _for_batch(batch_x, batch_t):
                    """
                    动态批处理：尽量用大批次训练；若显存不足触发 CUDA OOM，则对当前子批次二分，
                    直到能跑或无法再分。返回最后一次成功子批的 loss。
                    """
                    assert len(batch_x) == len(batch_t)
                    n = len(batch_x)
                    start = 0
                    last_loss = None

                    def _try_run(x, t):
                        optimizer.zero_grad(set_to_none=True)
                        if use_mixed_precision:
                            with torch.amp.autocast(device_type="cuda", dtype=precision_dtype):
                                out = self.forward(x)
                                logits = out["logits"]
                                targets = torch.tensor(t, dtype=torch.long, device=logits.device)  # [B]
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
                                sub_x = batch_x[start:end]
                                sub_t = batch_t[start:end]
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
                                    # 非 OOM 错误：直接抛出
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
                            batch_x = batchDataset[i:i + L]
                            batch_t = batchY[i:i + L]
                            lossItem = _for_batch(batch_x, batch_t)

                else:
                    for i in tqdm(range(0, len(batchDataset), batch)):
                        batch_x = batchDataset[i:i + batch]
                        batch_t = batchY[i:i + batch]
                        lossItem = _for_batch(batch_x, batch_t)

            else:
                for i in tqdm(range(len(dataset))):
                    for j in range(len(dataset[i][1])):
                        x = dataset[i][0] + dataset[i][1][:j]
                        y = dataset[i][1][j]

                        # out = model(x)
                        out = self.forward(x)
                        logits = out["logits"]
                        logits = logits.unsqueeze(0)

                        target = torch.tensor([y], dtype=torch.long, device=logits.device)

                        loss = criterion(logits, target)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    lossItem = loss.item()
            if use_temp_best:  # 保存最优模型
                self.temp_best_checkpoint = {
                    k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                }
            if save_model_path:
                if lossItem and lossItem < best_loss:
                    #  if loss.item() < best_loss:
                    best_loss = lossItem
                    self.save_model(
                        path=save_model_path,
                        optimizer=optimizer.state_dict(),
                        epoch=epoch,
                        loss=lossItem,
                    )
                    print(f"🔥 新最佳模型已保存 (loss={best_loss:.6f})")
            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {lossItem:.6f}, "
                      f"Time: {time.time() - startTime:.2f}s")

            break_loss = stop_loss
            if lossItem <= break_loss:
                print(f"低于:{break_loss},自动退出")
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
        }
        if optimizer:
            # 兼容 dict 或 优化器对象
            if isinstance(optimizer, dict):
                ckpt["optimizer_state_dict"] = optimizer
            else:
                ckpt["optimizer_state_dict"] = optimizer.state_dict()
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
            model.register_buffer("eid_table", model.build_lastid_to_eid_table())

        return model, ckpt

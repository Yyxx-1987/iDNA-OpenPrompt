# ---encoding:utf-8---
# @Time : 2021.05.05
# @Author : yuying0711
# @Email : 2539449171@qq.com
# @IDE : PyCharm



import torch
import torch.nn as nn
import numpy as np
from configuration import config
import pickle
from util import util_freeze
# from protlearn.features import aac
from collections import Counter
from BiLSTM_att import resnet_main, BiLSTM_main, Linear_full_main, performance
import matplotlib.pyplot as plt
import seaborn as sns
import Clip.clip as clip
from Cocoop import load_clip_to_cpu, CustomCLIP
def get_attn_pad_mask(seq):
    batch_size, seq_len = seq.size()
    pad_attn_mask = seq.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    pad_attn_mask_expand = pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]
    return pad_attn_mask_expand


class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding (look-up table) [64, 43, 64]
        self.pos_embed = nn.Embedding(max_len, d_model)  # position embedding [64, 43, 64]
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)  # x: [batch_size, seq_len]
        pos = torch.arange(seq_len, device=device, dtype=torch.long)  # [seq_len]
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        embedding = self.pos_embed(pos)
        embedding = embedding + self.tok_embed(x)
        embedding = self.norm(embedding)
        return embedding


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_head, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)  # [batch_size, n_head, seq_len, seq_len]
        context = torch.matmul(attn, V)  # [batch_size, n_head, seq_len, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_head)
        self.W_K = nn.Linear(d_model, d_k * n_head)
        self.W_V = nn.Linear(d_model, d_v * n_head)

        self.linear = nn.Linear(n_head * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # q_s: [batch_size, n_head, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_head, d_k).transpose(1, 2)  # k_s: [batch_size, n_head, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_head, d_v).transpose(1, 2)  # v_s: [batch_size, n_head, seq_len, d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_head, 1, 1)
        context, attention_map = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_head * d_v)  # context: [batch_size, seq_len, n_head * d_v]
        output = self.linear(context)
        output = self.norm(output + residual)
        return output, attention_map


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.attention_map = None

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attention_map = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                                        enc_self_attn_mask)  # enc_inputs to same Q,K,V
        self.attention_map = attention_map
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs, self.attention_map


class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()

        global max_len, n_layers, n_head, d_model, d_ff, d_k, d_v, vocab_size, device, N_CTX, CTX_INIT, PREC
        self.device = torch.device("cuda" if config.cuda else "cpu")
        max_len = config.max_len
        n_layers = config.num_layer
        n_head = config.num_head
        d_model = config.dim_embedding
        d_ff = config.dim_feedforward
        d_k = config.dim_k
        d_v = config.dim_v
        vocab_size = config.vocab_size
        device = torch.device("cuda" if config.cuda else "cpu")

        self.embedding = Embedding(config)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc_task = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),
        )
        # 20230531
        # hidden_size = 32
        hidden_size = 32
        self.lstm_hiden = 32
        self.max_seq_len = 41
        self.bilstm = nn.LSTM(hidden_size, self.lstm_hiden, 1, bidirectional=True, batch_first=True,
                              dropout=0.4)
        #
        self.classifier = nn.Linear(2, 2)
    #        20231106
        self.clipmodel, _ = clip.load("ViT-B/16", device=self.device, jit=False, return_intermediate_text_feature="0")
        config.N_CTX = 16
        config.CTX_INIT = ""
        config.PREC = "fp16"
        clip_model = load_clip_to_cpu(config)

        # if config.TRAINER.COCOOP.PREC == "fp32" or config.TRAINER.COCOOP.PREC == "amp":
        #     # CLIP's default precision is fp16
        clip_model.float()
        # config.TRAINER.COCOOP.N_CTX=
        # ctx_init = cfg.TRAINER.COCOOP.CTX_INIT


    def forward(self, input_ids):
        input_ids = input_ids.to(self.device)
        output = self.embedding(input_ids)  # [bach_size, seq_len, d_model]
        classnames = 'r_k_all'
        clip_model = load_clip_to_cpu(config)
        CustomCLIP_model = CustomCLIP(config, classnames, clip_model)
        CustomCLIP_model.cuda()
        # enc_self_attn_mask = get_attn_pad_mask(input_ids)  # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            # output = layer(output, enc_self_attn_mask)
            # 20231106
            print("input_ids", input_ids.shape)
            # print("output", output.shape)
            # output = self.clipmodel.encode_text(output, input_ids)
            # output = self.clipmodel(output, input_ids)
            # self.model = CustomCLIP(cfg, classnames, clip_model)
            output = CustomCLIP_model(output)
            # print("output2", output.shape)
            # 20231106
            # 20230913

            # output, attention = layer(output, enc_self_attn_mask)

            # attention = output[5, :, :]
            # attention *= 10
            # attention1 = attention.cpu()
            # # attention2 = attention1[16, 5, :, :]
            # # attention2 = np.random.rand(388, 388)
            # attention2 = np.random.rand(388, 448)
            # # attention2[:, :] = attention1[16, 5, :, :].detach().numpy()
            # attention2[:, :] = attention1[:, :].detach().numpy()
            # # attention3 = attention2.detach().numpy()
            # attention3 = attention2
            # # plt.imshow(attention3.squeeze().detach().numpy(), cmap='viridis', aspect='auto')
            # # plt.imshow(attention3, cmap='viridis', aspect='auto')
            # # sns.heatmap(attention3, cmap='viridis', square=True)
            # plt.imshow(attention3, cmap='YlGnBu', aspect='auto')
            # # plt.figure(figsize=(6, 8))
            # # plt.xlabel('输入序列位置')
            # # plt.ylabel('输入序列位置')
            # # plt.title('自注意力图')
            # # plt.colorbar()
            # plt.show()


            # #data enhane
            # attention = input_ids[:, :371]
            # plt.figure(figsize=(6, 2))  # 设置宽度为6英寸，高度为2英寸，根据需要调整高度
            # attention3 = attention.cpu()
            # # 绘制热力图，去掉标度尺
            # plt.imshow(attention3, cmap='YlGnBu', aspect='auto')
            # # plt.imshow(attention3, cmap='viridis', aspect='auto')
            # plt.colorbar().remove()  # 移除颜色条
            # plt.title('6mA_H.sapiens')
            #
            # # 去掉坐标轴
            # plt.axis('off')
            # plt.savefig('../model/6mA_H.sapiens_data enhance1.png', dpi=300, bbox_inches='tight')
            # plt.show()
            # # # 关闭图形窗口
            # # plt.close()
            #
            # # plt.savefig('../model/4mC_S.cerevisiae_data enhance1.png', dpi=300, bbox_inches='tight')
            # #data enhane
            # output: [batch_size, max_len, d_model]
        # classification
        # classification
        # only use [CLS]
        # # 20230531
        # seq_out = output  # [batchsize, max_len, 768]
        # batch_size = seq_out.size(0)
        # # batch_size = 32
        # seq_out, _ = self.bilstm(seq_out)
        # # seq_out = seq_out.contiguous().view(-1, self.lstm_hiden * 2)
        # seq_out = seq_out.contiguous().view(batch_size, -1, self.lstm_hiden)

        #
        representation = output[:, 0, :]
        # #
        # attention1 = representation.cpu()
        # # attention2 = attention1[0, 0, :, :]
        # plt.imshow(attention1.squeeze().detach().numpy(), cmap='viridis', aspect='auto')
        # plt.xlabel('输入序列位置')
        # plt.ylabel('输入序列位置')
        # plt.title('自注意力图')
        # plt.colorbar()
        # plt.show()
        # #

        # representation = seq_out[0, :, :]
        # 20230523________________
        # data_name = representation
        # batch_size = 16
        # epochs = 40
        # learning_rate = 1e-4
        # best_acc = 0.0
        # representation = BiLSTM_main(data_name, epochs, batch_size, learning_rate, best_acc)

        # 20230523___________________

        reduction_feature = self.fc_task(representation)
        reduction_feature = reduction_feature.view(reduction_feature.size(0), -1)
        logits_clsf = self.classifier(reduction_feature)
        representation = reduction_feature
        return logits_clsf, representation

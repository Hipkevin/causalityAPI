import torch
import torch.nn as nn
import math
from torchcrf import CRF


class BiLstmCRF(nn.Module):
    def __init__(self, config):
        super(BiLstmCRF, self).__init__()

        # embedding随训练更新
        self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)

        self.device = config.device

        self.tag_num = config.tag_num
        self.seqLen = config.pad_size

        self.lstm = nn.LSTM(input_size=config.input_size,
                            hidden_size=config.hidden_size // 2,
                            num_layers=config.hidden_layer_num,
                            dropout=config.dropout,
                            bidirectional=True,  # 设置双向
                            batch_first=True)  # batch为张量的第一个维度

        self.attention = SelfAttention(config.pad_size)

        self.pos_fc = PositionFeedForward(config.pad_size, config.pad_size)

        self.conditionLinear = nn.Linear(config.hidden_size, config.pad_size)
        self.attLinear = nn.Linear(config.pad_size, 1)
        self.linear = nn.Linear(config.pad_size, config.tag_num)

        self.conditionNorm = ConditionalLayerNorm(config.pad_size)

        self.crf = CRF(config.tag_num, batch_first=True)

    def forward(self, x, pos):
        # embedding&预测层
        out = self.embedding(x)
        out, _ = self.lstm(out)

        pos = self.pos_fc(pos)

        out = self.conditionLinear(out)
        out = self.conditionNorm(out, pos)

        # out = self.attention(out)
        # out = torch.tanh(out)
        sentence = torch.squeeze(self.attLinear(out))
        att = torch.unsqueeze(self.attention(sentence), 2)

        out = out * att

        # 维度线性变换
        out = self.linear(out)

        # crf解码
        # 计算loss时使用crf的前向传播
        tag = self.crf.decode(out)

        return out, tag

    def loss_func(self, out, y, mask):
        weight = torch.tensor([1] * self.tag_num, dtype=torch.float32).to(self.device)
        weight[0] = 0.08
        weight[2:12:2] = 0.63

        # weight[5] = .0
        # weight[6] = .0

        weight[11] = .0
        weight[12] = .0

        criterion = nn.CrossEntropyLoss(weight=weight)

        crf_loss = - self.crf(out, y, mask, 'token_mean')  # crf-loss

        lstm_loss = torch.tensor(0, dtype=torch.float32).to(self.device)
        for Y_hat, Y in zip(out, y):
            lstm_loss += criterion(Y_hat, Y)

        return crf_loss, lstm_loss


class PositionFeedForward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(PositionFeedForward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out

class ConditionalLayerNorm(nn.Module):
    def __init__(self,
                 normalized_shape,
                 eps=1e-12):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(normalized_shape))
        self.bias = nn.Parameter(torch.Tensor(normalized_shape))

        self.weight_dense = nn.Linear(normalized_shape, normalized_shape, bias=False)
        self.bias_dense = nn.Linear(normalized_shape, normalized_shape, bias=False)

        self.reset_weight_and_bias()

    def reset_weight_and_bias(self):
        """
        此处初始化的作用是在训练开始阶段不让 conditional layer norm 起作用
        """
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

        nn.init.zeros_(self.weight_dense.weight)
        nn.init.zeros_(self.bias_dense.weight)

    def forward(self, inputs, cond=None):
        assert cond is not None, 'Conditional tensor need to input when use conditional layer norm'
        cond = torch.unsqueeze(cond, 1)  # (b, 1, h*2)

        weight = self.weight_dense(cond) + self.weight  # (b, 1, h)
        bias = self.bias_dense(cond) + self.bias  # (b, 1, h)

        mean = torch.mean(inputs, dim=-1, keepdim=True)  # （b, s, 1）
        outputs = inputs - mean  # (b, s, h)

        variance = torch.mean(outputs ** 2, dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.eps)  # (b, s, 1)

        outputs = outputs / std  # (b, s, h)

        outputs = outputs * weight + bias

        return outputs

class SelfAttention(nn.Module):
    def __init__(self, n_dim):
        super(SelfAttention, self).__init__()
        self.n_dim = n_dim
        self.W_q = nn.Linear(n_dim, n_dim)
        self.W_k = nn.Linear(n_dim, n_dim)
        self.W_v = nn.Linear(n_dim, n_dim)

        self.norm = nn.LayerNorm(n_dim)

        self.scale = math.sqrt(n_dim)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        energy = Q * K / self.scale

        alpha = torch.softmax(energy, dim=-1)

        x = self.norm(alpha * V)

        return x
import torch
import numpy as np

class ModelConfig:
    def __init__(self):
        # embedding
        self.vocab_path = "util/vocab.pkl"
        self.embedding_pretrained = torch.tensor(
            np.load("util/embedding_sgns_financial.npz")["embeddings"].astype('float32'))

        self.pad_size = 512  # 填充长度
        self.tag_num = 13  # 标签集合大小

        # lstm
        self.input_size = self.embedding_pretrained.size(1)
        self.hidden_size = self.input_size
        self.hidden_layer_num = 5
        self.dropout = 0.2

        # param
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.batch_size = 32
        self.epoch = 5

        self.learning_rate = 1e-3
        self.crf_lr = 1e-2

        self.decay = 1e-4
        self.crf_decay = 1e-5

        self.T_0 = 5
        self.T_mult = 2

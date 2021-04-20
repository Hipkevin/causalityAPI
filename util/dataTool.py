import json
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import pickle as pkl


""" 常量 """
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
TAG = {'原因中的核心名词': 'CauseN',
       '原因中的谓语或状态': 'CauseV',
       '中心词': 'trigger',
       '结果中的核心名词': 'ResN',
       '结果中的谓语或状态': 'ResV'}

TAG_DICT = {"O": 0,
            "B-CauseN": 1,
            "I-CauseN": 2,
            "B-CauseV": 3,
            "I-CauseV": 4,

            "B-trigger": 5,
            "I-trigger": 6,

            "B-ResN": 7,
            "I-ResN": 8,
            "B-ResV": 9,
            "I-ResV": 10,
            "<UNK>": 11,
            "<PAD>": 12}

def processJson(json_dict, vocab, config):
    UNK, PAD = '<UNK>', '<PAD>'

    document = json_dict['document'][0]
    text = document['text']
    seqLen = len(text)
    token = list(text)
    mask = [1] * seqLen
    tag = list()

    pad_size = config.pad_size

    for item in json_dict['qas'][0]:
        for i in item['answers']:
            tag.append(i['text'])

    trigger_dict = json_dict['qas'][0][2]
    trigger_start = int(trigger_dict['answers'][0]['start'])
    trigger_end = int(trigger_dict['answers'][0]['end'])

    position = np.arange(seqLen)
    position[0: trigger_start + 1] = position[0: trigger_start + 1] - trigger_start
    position[trigger_end:] = position[trigger_end:] - trigger_end
    position[trigger_start: trigger_end + 1] = 0
    position = list(position)

    if len(token) < pad_size:
        mask.extend([0] * (pad_size - len(token)))
        position.extend([1000] * (pad_size - len(token)))
        token.extend([PAD] * (pad_size - len(token)))
    else:
        mask = mask[: pad_size]
        position = position[: pad_size]
        token = token[: pad_size]

    # word to id
    words_id = list()
    for word in token:
        words_id.append(vocab.get(word, vocab.get(UNK)))

    return words_id, mask, position, tag, text


def tokenize(vocab, text, config):
    tokenizer = lambda x: [y for y in x]  # char-level
    token = tokenizer(text)

    pad_size = config.pad_size

    if len(token) < pad_size:
        token.extend([PAD] * (pad_size - len(token)))
    else:
        token = token[: pad_size]

    # word to id
    words_id = list()
    for word in token:
        words_id.append(vocab.get(word, vocab.get(UNK)))

    return words_id

def build_dataset(config):
    """
    构建数据集
    :param config: 配置文件
    :return: 返回词表及X(texts, masks)，Y(tags, seqLen)
    """
    tokenizer = lambda x: [y for y in x]  # char-level

    vocab = pkl.load(open(config.vocab_path, 'rb'))  # 加载词表

    # 闭包
    def load_dataset(path, pad_size=32):
        with open(path, encoding='utf8') as file:
            data = file.readlines()

        contents = list()
        tags = list()
        masks = list()
        positions = list()
        for d in data:
            dataDict = json.loads(d)

            # json解析后抽取数据元素
            document = dataDict['document'][0]
            text = document['text']
            seqLen = len(text)
            tagSeq = ['O'] * seqLen
            mask = [1] * seqLen

            # 生成标签序列
            for item in dataDict['qas'][0]:
                question = item['question']
                answers = item['answers']
                if answers:
                    for ans in answers:
                        start = ans['start']
                        end = ans['end']

                        tagSeq[start] = 'B-' + TAG[question]
                        tagSeq[start + 1: end + 1] = ['I-' + TAG[question]] * (end - start)
                        # mask[start: end + 1] = [1] * (end - start + 1)

            # trigger位置标记
            position = np.arange(seqLen)
            tagSeqArr = np.array(tagSeq, dtype=np.str)

            trigger_start = np.where(tagSeqArr == 'B-trigger')[0][0]

            last_trigger = np.where(tagSeqArr == 'I-trigger')[0]
            if last_trigger.any():
                trigger_end = last_trigger[-1]
            else:
                trigger_end = trigger_start

            position[0: trigger_start + 1] = position[0: trigger_start + 1] - trigger_start
            position[trigger_end:] = position[trigger_end:] - trigger_end
            position[trigger_start: trigger_end + 1] = 0
            position = list(position)

            # padding对齐
            token = tokenizer(text)
            if len(token) < pad_size:
                tagSeq.extend(["O"] * (pad_size - len(token)))
                mask.extend([0] * (pad_size - len(token)))
                position.extend([1000] * (pad_size - len(token)))
                token.extend([PAD] * (pad_size - len(token)))
            else:
                token = token[: pad_size]
                tagSeq = tagSeq[: pad_size]
                mask = mask[: pad_size]
                position = position[: pad_size]
                seqLen = len(token)

            # word to id
            words_id = list()
            for word in token:
                words_id.append(vocab.get(word, vocab.get(UNK)))

            # tag to id
            tag_id = list()
            for tag in tagSeq:
                tag_id.append((TAG_DICT.get(tag), seqLen))

            contents.append(words_id)
            tags.append(tag_id)
            masks.append(mask)
            positions.append(position)

        return (contents, masks, positions), tags

    X, Y = load_dataset(config.train_path, config.pad_size)

    return vocab, X, Y


class TagDataSet(Dataset):
    """重写生成器"""
    def __init__(self, X, Y):
        super(TagDataSet, self).__init__()

        self.data, self.mask, self.pos = X[0], X[1], X[2]
        self.tag = Y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.mask[index], self.pos[index], self.tag[index]
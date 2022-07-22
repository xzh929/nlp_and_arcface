import jieba
from utils import word2id, list_word2id
from torch import nn
import torch
from torch.utils.data import Dataset


text = open(r"../nlp/text.txt", encoding="utf-8", mode="r")
lines = text.readlines()
text_list = []
embedding = nn.Embedding(125, 3, padding_idx=0)

for line in lines:
    line = line.strip()
    line = jieba.lcut(line)
    text_list.append(line)
text.close()
maps = word2id(text_list)
list_id = list_word2id(text_list)
list_emb = embedding(torch.tensor(list_id))


class Sendataset(Dataset):
    def __init__(self):
        self.sentences_list = list_id
        self.sentences_emb_weight = torch.randn(125, 3)
        self.ori_data = []
        for sentence in self.sentences_list:
            for i, word in enumerate(sentence):
                if i == 0:
                    self.ori_data.append([word, [0, 0, sentence[i + 1], sentence[i + 2]]])
                elif i == 1:
                    self.ori_data.append([word, [0, sentence[i - 1], sentence[i + 1], sentence[i + 2]]])
                elif i == len(sentence) - 2:
                    self.ori_data.append([word, [sentence[i - 2], sentence[i - 1], sentence[i + 1], 0]])
                elif i == len(sentence) - 1:
                    self.ori_data.append([word, [sentence[i - 2], sentence[i - 1], 0, 0]])
                else:
                    self.ori_data.append([word, [sentence[i - 2], sentence[i - 1], sentence[i + 1], sentence[i + 2]]])

    def __len__(self):
        return len(self.ori_data)

    def __getitem__(self, item):
        return torch.tensor(self.ori_data[item][0]), torch.tensor(self.ori_data[item][1])


if __name__ == '__main__':
    dataset = Sendataset()
    print(len(dataset))
    data, label = dataset[0]
    print(data, label)

import random
import numpy as np
import torch
from torch import nn


# 带批次的word2id
def word2id(batch_list):
    maps = {}
    for list in batch_list:
        for item in list:
            if item not in maps:
                maps[item] = len(maps) + 1
    maps['eos'] = 124
    maps['pad'] = 0
    return maps


def list_word2id(batch_list):
    maps = word2id(batch_list)
    max_length = len(max(batch_list, key=len))
    for list in batch_list:
        list_length = len(list)
        if list_length < max_length:
            for i in range(max_length - list_length):
                list.append('pad')
        for i, item in enumerate(list):
            list[i] = maps[item]
        list.append(maps['eos'])
    return batch_list


def swapPositions(list, pos1, pos2):
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list


def endswapstart(list):
    end = list[len(list) - 1]
    list.pop()
    list.insert(0, end)
    return list


def getvaluedim2(tensor, tag):
    idx = torch.nonzero(tag)
    return tensor[idx[:, 0], idx[:, 1]]


if __name__ == '__main__':
    feature = torch.randn(5, 2)
    tag = torch.tensor([1, 1, 0, 1, 0])

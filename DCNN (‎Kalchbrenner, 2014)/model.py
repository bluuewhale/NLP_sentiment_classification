from collections import defaultdict
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import numpy as np

class CnnText(nn.Module):

    def __init__(self, n_words, embed_size, hid_size, drop_rate, kernel_size_ls, num_filter):
        super(CnnText, self).__init__()

        self.embed_size = embed_size
        self.hid_size = hid_size
        self.drop_rate = drop_rate
        self.num_filter = num_filter
        self.kernel_size_ls = kernel_size_ls
        self.num_kernel = len(kernel_size_ls)

        self.embedding = nn.Embedding(n_words, embed_size)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filter, (kernel_size, embed_size)) for kernel_size in kernel_size_ls])

        self.lin = nn.Sequential(
            nn.Linear(self.num_kernel*num_filter, hid_size), nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(hid_size, 1),
            )

    def forward(self, x):
        embed = self.embedding(x) # [batch_size, max_length, embed_size]
        embed.unsqueeze_(1)  # [batch_size, 1, max_length, embed_size]
        conved = [conv(embed).squeeze(3) for conv in self.convs] # [batch_size, num_filter, max_length -kernel_size +1]
        pooled = [F.max_pool1d(conv, (conv.size(2))).squeeze(2) for conv in conved] # [batch_size, num_kernel, num_filter]
        concated = torch.cat(pooled, dim = 1) # [batch_size, num_kernel * num_filter]
        logit = self.lin(concated)

        return torch.sigmoid(logit)

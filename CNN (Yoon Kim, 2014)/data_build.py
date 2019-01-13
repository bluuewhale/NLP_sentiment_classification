# data는 e9t(Lucy Park)님께서 github에 공유해주신 네이버 영화평점 데이터를 사용하였습니다.
# https://github.com/e9t/nsmc

from collections import defaultdict
import torch
from torch.autograd import Variable


class DataSetUp(object):

    def __init__(self):
        self.w2i_dict = defaultdict(lambda : len(self.w2i_dict))
        self.pad = self.w2i_dict['<PAD>']

    def read_txt(self,path_to_file):
        txt_ls = []
        label_ls = []

        with open(path_to_file) as f:
            for i, line in enumerate(f.readlines()[1:]):
                id_num, txt, label = line.split('\t')
                txt_ls.append(txt)
                label_ls.append(int(label.replace('\n','')))
        return txt_ls, label_ls


    def convert_word_to_idx(self,sents):
        for sent in sents:
            yield [self.w2i_dict[word] for word in sent.split(' ')]
        return

    def add_padding(self, sents, max_len):
        for i, sent in enumerate(sents):
            if len(sent)< max_len:
                sents[i] += [self.pad] * (max_len - len(sent))

            elif len(sent) > max_len:
                sents[i] = sent[:max_len]
        return sents

    def convert_to_variable(self, sents):
        var = Variable(torch.LongTensor(sents))
        return var

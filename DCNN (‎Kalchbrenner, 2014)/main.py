# data는 e9t(Lucy Park)님께서 github에 공유해주신 네이버 영화평점 데이터를 사용하였습니다.
# https://github.com/e9t/nsmc


from data_build import DataSetUp
from model import CnnText

from collections import defaultdict
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import numpy as np


def main():
    data_setup = DataSetUp()

    # 데이터 불러오기
    train_txt_ls, train_label_ls = data_setup.read_txt('ratings_train.txt')
    test_txt_ls, test_label_ls = data_setup.read_txt('ratings_test.txt')

    # str을 w2i로 변환
    train_w2i_ls = list(data_setup.convert_word_to_idx(train_txt_ls))
    test_w2i_ls = list(data_setup.convert_word_to_idx(test_txt_ls))

    i2w_dict = {val : key for key, val in data_setup.w2i_dict.items()}

    # pytorch 모델 학습을 위한 데이터 build_up
    x_train = data_setup.convert_to_variable(data_setup.add_padding(train_w2i_ls, 15))
    x_val = data_setup.convert_to_variable(data_setup.add_padding(test_w2i_ls[:10000],15))
    x_test = data_setup.convert_to_variable(data_setup.add_padding(test_w2i_ls[10000:],15))

    y_train = data_setup.convert_to_variable(train_label_ls).float()
    y_val = data_setup.convert_to_variable(test_label_ls[:10000]).float()
    y_test = data_setup.convert_to_variable(test_label_ls[10000:]).float()

    # model hypterparameter
    n_words = len(data_setup.w2i_dict)
    EMBED_SIZE = 64
    HID_SIZE = 64
    DROP_RATE = 0.5
    KERNEL_SIZE_LS = [2,3,4,5]
    NUM_FILTER = 16

    # model
    model = CnnText(
        n_words = n_words,
        embed_size =EMBED_SIZE,
        drop_rate= DROP_RATE,
        hid_size=HID_SIZE,
        kernel_size_ls= KERNEL_SIZE_LS,
        num_filter=NUM_FILTER
        )


    '''
    TRAIN
    '''
    epochs = 30
    lr = 0.001
    batch_size = 10000

    train_idx = np.arange(x_train.size(0))
    test_idx = np.arange(x_test.size(0))
    optimizer = torch.optim.Adam(model.parameters(),lr)
    criterion = nn.BCELoss(reduction='sum')

    for epoch in range(epochs):
        model.train()

        # input 데이터 순서 섞기
        random.shuffle(train_idx)
        x_train = x_train[train_idx]
        y_train = y_train[train_idx]
        train_loss = 0

        for start_idx, end_idx in zip(range(0, x_train.size(0), batch_size),
                                      range(batch_size, x_train.size(0)+1, batch_size)):
            x_batch = x_train[start_idx : end_idx]
            y_batch = y_train[start_idx : end_idx]

            logit = model(x_batch)
            predict = logit.ge(0.5).float()
            y_batch = y_batch.unsqueeze(1)

            acc = (predict == y_batch).sum().item() / batch_size
            loss = criterion(logit, y_batch)
            train_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Train epoch : %s,  loss : %s,  accuracy :%.3f'%(epoch+1, train_loss.item(), acc))
        print('=================================================================================================')

        if (epoch+1) % 10 == 0:
            model.eval()
            logit = model(x_val).squeeze(1)
            predict = logit.ge(0.5).float()

            acc = (predict == y_val).sum().item() / 10000
            loss = criterion(logit, y_val)
            print('******************************************************************************************************************')
            print('*******************   Test Epoch : %s, Test Loss : %.03f , Test Accuracy : %.03f    *******************'%(epoch+1, loss.item(), acc))
            print('******************************************************************************************************************')

if __name__=='__main__':
    main()

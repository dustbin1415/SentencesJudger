import os
import torch
import torch.nn as nn
import jieba.posseg as pseg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 31
HIDDEN_SIZE = 256
RNN_LAYER = 2
BIDIRECTIONAL = True
BATCH_FIRST = True
model_path = '../models/model.pkl'
hidden = None

class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.hidden_size = HIDDEN_SIZE
        self.gru_layers = RNN_LAYER
        self.gru = nn.GRU(
        input_size = INPUT_SIZE,
        hidden_size = HIDDEN_SIZE,
        num_layers = RNN_LAYER,
        bidirectional = BIDIRECTIONAL,
        batch_first = BATCH_FIRST,
        )
        if BIDIRECTIONAL:
            self.out = nn.Linear(HIDDEN_SIZE * 2, 2)
        else:
            self.out = nn.Linear(HIDDEN_SIZE, 2)

    
    def forward(self, input, hidden):
        t_out, hidden = self.gru(input, hidden)
        if BIDIRECTIONAL:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim = 1)
        else:
            hidden_cat = hidden[-1]
        return self.out(hidden_cat)

def Run(input):
    output = gru(torch.FloatTensor([input]), hidden)
    if output[0][0] > output[0][1]: 
        tag = True
    else: 
        tag = False
    return tag

if __name__ == '__main__':
    try:
        print('加载中...')
        gru = torch.load(model_path,map_location='cpu')
        gru = gru.to(DEVICE)
        print('加载完成')
        dist = {'a': 0, 'ad': 1, 'ag': 2, 'an': 3, 'b': 4, 'c': 5, 'd': 6, 'df': 7, 'dg': 8, 'e': 9, 
            'f': 10, 'g': 11, 'h': 12, 'i': 13, 'j': 14, 'k': 15, 'l': 16, 'm': 17, 'mg': 18,
            'mq': 19, 'n': 20, 'ng': 21, 'nr': 22, 'nrfg': 23, 'nrt': 24, 'ns': 25, 'nt': 26,
            'nz': 27, 'o': 28, 'p': 29, 'q': 30, 'r': 31, 'rg': 32, 'rr': 33, 'rz': 34,
            's': 35, 't': 36, 'tg': 37, 'u': 38, 'ud': 39, 'ug': 40, 'uj': 41, 'ul': 42,
            'uv': 43, 'uz': 44, 'v': 45, 'vd': 46, 'vg': 48, 'vi': 49, 'vn': 50, 'vq': 51,
            'y': 52, 'yg': 53, 'z': 54, 'zg': 55, 'eng': 56}
        while True:
            string = input('请输入一个句子:')
            tmp = pseg.cut(string, use_paddle=True)
            words = []
            for word, flag in tmp:
                if(flag != 'w' and flag != 'x'):
                    tmp_list = list(word.encode())
                    words += [tmp_list + [0 for i in range(30 - len(tmp_list))] + [dist[flag]]]
            if Run(words):
                print("True")
            else:
                print("False")
    except:
        print("加载失败")
        os.system('pause')

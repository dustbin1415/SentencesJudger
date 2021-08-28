import os
import torch
import torch.nn as nn
from ltp import LTP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 30
HIDDEN_SIZE = 256
RNN_LAYER = 2
BIDIRECTIONAL = True
BATCH_FIRST = True
model_path = './model.pkl'
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
    for word in input:
        output = gru(torch.FloatTensor([[word]]), hidden)
    if output[0][0] > output[0][1]: 
        tag = True
    else: 
        tag = False
    return tag

if __name__ == '__main__':
    try:
        print('加载中...')
        gru = torch.load(model_path)
        gru = gru.to(DEVICE)
        ltp = LTP()
        print('加载完成')
        while True:
            string = input('请输入一个句子:')
            seg, tmp = ltp.seg([string])
            words = []
            for word in seg[0]:
                tmp_list = list(word.encode())
                words += [tmp_list + [0 for i in range(30 - len(tmp_list))]]
            if Run(words):
                print("True")
            else:
                print("False")
    except:
        print("加载失败")
        os.system('pause')
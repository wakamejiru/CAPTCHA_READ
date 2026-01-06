import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, img_h, nc, n_class, nh, dropout=0.5):
        """
        img_h: 入力画像の高さ (32)
        nc: 入力チャンネル数 (3)
        n_class: クラス数 (36 + 1)
        nh: LSTMの隠れ層サイズ (512)
        dropout: ドロップアウト率
        """
        super(CRNN, self).__init__()

        # --- CNN Part (VGG Style) ---
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2)) 
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        convRelu(6, True)
        cnn.add_module('pooling{0}'.format(4), nn.AdaptiveMaxPool2d((1, None)))

        self.cnn = cnn

        # --- RNN Part (BiLSTM + Dropout) ---
        rnn_input_size = nm[-1] 
        
        self.rnn = nn.Sequential(
            BidirectionalLSTM(rnn_input_size, nh, nh),
            nn.Dropout(dropout),
            # 【修正点】入力サイズを nh * 2 から nh に変更しました
            BidirectionalLSTM(nh, nh, n_class)
        )

    def forward(self, input):
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, f"the height of conv must be 1, but got {h}"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [Width, Batch, Channel]
        output = self.rnn(conv)
        return output

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output
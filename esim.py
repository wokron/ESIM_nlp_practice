import torch
from torch import nn
from torch.nn import functional as F


class ESIM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size1, hidden_size2, num_layers, class_num):
        super(ESIM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm1 = nn.LSTM(embed_size, hidden_size1, num_layers, bidirectional=True, batch_first=True)

        self.lstm2 = nn.LSTM(8 * hidden_size1, hidden_size2, num_layers, bidirectional=True, batch_first=True)

        self.multi = nn.Sequential(
            nn.Linear(8 * hidden_size2, hidden_size2),
            nn.Tanh(),
            nn.Linear(hidden_size2, class_num),
        )

    def forward(self, s1, s2):
        a = self.embed(s1)  # (batch_size,seq_size1,embed_size)
        b = self.embed(s2)  # (batch_size,seq_size2,embed_size)

        a_, _ = self.lstm1(a)  # (batch_size,seq_size1,2*hidden_size)
        b_, _ = self.lstm1(b)  # (batch_size,seq_size2,2*hidden_size)

        e = torch.bmm(a_, b_.permute(0, 2, 1))  # (batch_size,seq_size1,seq_size2)

        a_sim = torch.bmm(F.softmax(e, dim=-1), b_)  # (batch_size,seq_size1,2*hidden_size)
        b_sim = torch.bmm(F.softmax(e.permute(0, 2, 1), dim=-1), a_)  # (batch_size,seq_size2,2*hidden_size)

        ma = torch.cat([a_, a_sim, a_ - a_sim, a_ * a_sim], dim=2)  # (batch_size,seq_size1,8*hidden_size)
        mb = torch.cat([b_, b_sim, b_ - b_sim, b_ * b_sim], dim=2)  # (batch_size,seq_size2,8*hidden_size)

        va, _ = self.lstm2(ma)  # (batch_size,seq_size1,2*hidden_size2)
        vb, _ = self.lstm2(mb)  # (batch_size,seq_size2,2*hidden_size2)

        va_ave = torch.mean(va, dim=1)  # (batch_size,2*hidden_size2)
        va_max, _ = torch.max(va, dim=1)  # (batch_size,2*hidden_size2)

        vb_ave = torch.mean(vb, dim=1)  # (batch_size,2*hidden_size2)
        vb_max, _ = torch.max(vb, dim=1)  # (batch_size,2*hidden_size2)

        v = torch.cat([va_ave, va_max, vb_ave, vb_max], dim=1)  # (batch_size,8*hidden_size2)

        predict = self.multi(v)  # (batch_size,class_num)

        return predict

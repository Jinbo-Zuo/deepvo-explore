
import torch
import torch.nn as nn
from parameters import par
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_
import numpy as np

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.ReLU(),
        )

def fc(in_planes, out_planes, activation=False):
    if activation:
        return nn.Sequential(
            nn.Linear(in_planes, out_planes),
            nn.ReLU(),
        )
    else:
        return nn.Linear(in_planes, out_planes)

class DeepVO(nn.Module):
    def __init__(self, imsize_h, imsize_w, batchNorm=True):
        super(DeepVO,self).__init__()
        # CNN
        self.batchNorm = batchNorm
        self.conv1   = conv(self.batchNorm,    6,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,   64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm,  128,  256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm,  256,  256, kernel_size=3, stride=1)
        self.conv4   = conv(self.batchNorm,  256,  512, kernel_size=3, stride=2)
        self.conv4_1 = conv(self.batchNorm,  512,  512, kernel_size=3, stride=1)
        self.conv5   = conv(self.batchNorm,  512,  512, kernel_size=3, stride=2)
        self.conv5_1 = conv(self.batchNorm,  512,  512, kernel_size=3, stride=1)
        self.conv6   = conv(self.batchNorm,  512, 1024, kernel_size=3, stride=2)

        # self.conv6_1 = conv(self.batchNorm, 1024, 1024)
        # self.pool_1  = nn.MaxPool2d(2, stride=2)
        # self.dropout1 = nn.Dropout(0.5)
        # self.fc_1    = fc(1024 * 3 * 10, 4096, activation=True)
        # self.dropout2 = nn.Dropout(0.5)
        # self.fc_2    = fc(4096, 1024, activation=True)
        # self.fc_3    = fc(1024, 128, activation=True)
        # self.fc_4    = fc(128, 6)

        # Comput the shape based on diff image size
        __tmp = Variable(torch.zeros(1, 6, imsize_h, imsize_w))
        __tmp = self.encode_image(__tmp)

        # RNN
        self.rnn = nn.LSTM(
                    input_size=int(np.prod(__tmp.size())), 
                    hidden_size=par.rnn_hidden_size, 
                    num_layers=2, 
                    dropout=par.rnn_dropout_between, 
                    batch_first=True)
        self.rnn_drop_out = nn.Dropout(par.rnn_dropout_out)
        self.linear = nn.Linear(in_features=par.rnn_hidden_size, out_features=6)

        # Initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LSTM):
                # layer 1
                kaiming_normal_(m.weight_ih_l0)  #orthogonal_(m.weight_ih_l0)
                kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                # Set forget gate bias to 1 (remember)
                n = m.bias_hh_l0.size(0)
                start, end = n//4, n//2
                m.bias_hh_l0.data[start:end].fill_(1.)

                # layer 2
                kaiming_normal_(m.weight_ih_l1)  #orthogonal_(m.weight_ih_l1)
                kaiming_normal_(m.weight_hh_l1)
                m.bias_ih_l1.data.zero_()
                m.bias_hh_l1.data.zero_()
                n = m.bias_hh_l1.size(0)
                start, end = n//4, n//2
                m.bias_hh_l1.data[start:end].fill_(1.)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x): 
        # x: (batch, seq_len, channel, width, height)
        # stack_image
        x = torch.cat(( x[:, :-1], x[:, 1:]), dim=2)
        batch_size = x.size(0)
        seq_len = x.size(1)
        # CNN
        x = x.view(batch_size*seq_len, x.size(2), x.size(3), x.size(4))
        x = self.encode_image(x)
        x = x.view(batch_size, seq_len, -1)

        # RNN
        out, hc = self.rnn(x)
        out = self.rnn_drop_out(out)
        out = self.linear(out)
        return out

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def get_loss(self, x, y):
        predicted = self.forward(x)
        y = y[:, 1:, :]  # (batch, seq, dim_pose)
        # Weighted MSE Loss
        angle_loss = torch.nn.functional.mse_loss(predicted[:,:,:3], y[:,:,:3])
        translation_loss = torch.nn.functional.mse_loss(predicted[:,:,3:], y[:,:,3:])
        loss = (100 * angle_loss + translation_loss)
        return loss

    def step(self, x, y, optimizer):
        optimizer.zero_grad()
        loss = self.get_loss(x, y)
        loss.backward()
        optimizer.step()
        return loss

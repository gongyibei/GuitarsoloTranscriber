from turtle import forward
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from transformer.Layers import EncoderLayer
from transformer.Models import PositionalEncoding
from utils import plot_result


class Encoder(nn.Module):
    def __init__(self,
                 n_layers,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 dropout=0.1,
                 n_position=200,
                 scale_emb=False):
        super(Encoder, self).__init__()
        self.position_enc = PositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # self.activate = nn.ReLU()
        self.activate = nn.Tanh()
        # self.activate = nn.Sigmoid()
        self.scale_emb = scale_emb

    def forward(self, X):
        X = self.position_enc(X)
        if self.scale_emb:
            X *= self.d_model**0.5
        X = self.dropout(X)
        X = self.layer_norm(X)
        for enc_layer in self.layer_stack:
            X, att = enc_layer(X)
            X = self.activate(X)
        return X


class Decoder(nn.Module):
    def __init__(self, d_model, n_frame):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True)
        self.linear = nn.Linear(n_frame, 1)

    def forward(self, X):
        X = self.lstm(X)[0]
        # X = F.tanh(X)
        X = X.transpose(1, 2)
        X = self.linear(X)
        X = X.transpose(1, 2)
        
        # return F.sigmoid(X)
        return X


class GuitarFormer(pl.LightningModule):
    def __init__(
        self,
        d_feat=84,
        n_frame=4,
    ):
        super(GuitarFormer, self).__init__()
        self.d_feat = d_feat
        self.n_frame = n_frame
        self.encoder = Encoder(n_layers=4,
                               d_model=d_feat,
                               d_inner=d_feat * 2,
                               n_head=7,
                               d_k=12,
                               d_v=12,
                               dropout=0.1,
                               n_position=200,
                               scale_emb=False)
        self.decoder = Decoder(d_model=d_feat, n_frame=n_frame)
        # self.metric = nn.MSELoss()
        self.metric = nn.HuberLoss()
        # self.metric = nn.BCELoss()
        self.example_input_array = torch.randn(1, 4, 84)

    def forward(self, X):
        X = self.encoder(X)
        X = self.decoder(X)
        return F.sigmoid(X)

    def loss(self, out, y):
        out = out.reshape(-1)
        y = y.reshape(-1)

        # loss1 for notes not played
        loss1 = self.metric(out, y)

        # loss2 for notes being played
        idx = torch.where(y==1)[0]
        loss2 = self.metric(out[idx], y[idx])
        
        # loss3 for octave note
        # oct_idx = torch.hstack([(idx%12).unique() + 12*i for i in range(7)])
        # oct_out = out[oct_idx]
        # oct_y = y[oct_idx]
        # idx = torch.where(oct_y==0)[0]
        # loss3 = self.metric(oct_out[idx], oct_y[idx])

        # oct_idx = torch.hstack([idx-12, idx+12])
        oct_idx = torch.hstack([idx+12])
        oct_idx = oct_idx[torch.where((oct_idx>=0) & (oct_idx<84))[0]].unique()
        # print(oct_idx)
        oct_out = out[oct_idx]
        oct_y = y[oct_idx]
        idx = torch.where(oct_y==0)[0]
        loss3 = self.metric(oct_out[idx], oct_y[idx])
        loss3[loss3!=loss3] = 0

        # print(loss1, loss2, loss3)

        return loss1, loss2*0.1, loss3*0.1
    
    def binary_loss(self, out, y):
        out = out.reshape(-1)
        y = y.reshape(-1)
        loss = self.metric(out, y)
        return loss
        

    def training_step(self, batch, batch_idx):
        # forward
        x, y = batch
        # x, y = x[:, :self.n_frame, :], y[:, self.n_frame - 1:self.n_frame, :]
        x, y = x[:, :self.n_frame, :], y[:, self.n_frame - 1:self.n_frame, :]
        out = self(x)

        # calculate loss
        loss1, loss2, loss3 = self.loss(out, y)
        loss = loss1 + loss2
        # print(loss)

        # update learning rate
        sch = self.lr_schedulers()
        if batch_idx % 100 == 0:
            sch.step()

        # logging to tensorboard
        self.log('train_loss1', loss1, on_step=True, on_epoch=False)
        self.log('train_loss2', loss2, on_step=True, on_epoch=False)
        self.log('train_loss3', loss3, on_step=True, on_epoch=False)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        # forward
        x, y = batch
        x, y = x[:, :self.n_frame, :], y[:, self.n_frame - 1:self.n_frame, :]
        out = self(x)

        y = y.reshape((-1, y.shape[2]))
        out = out.reshape((-1, out.shape[2]))
        image = plot_result(out.detach().cpu().numpy().T,
                            y.detach().cpu().numpy().T)

        grid = torchvision.utils.make_grid(torch.tensor(image))
        self.logger.experiment.add_image('images', grid, batch_idx)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=1e-3)
        sch = optim.lr_scheduler.ExponentialLR(opt, 0.99)
        return [opt], [sch]


if __name__ == '__main__':
    model = GuitarFormer()
    data = torch.randn((100, 4, 84))
    print(model(data).shape)
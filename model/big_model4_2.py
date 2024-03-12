import torch.nn as nn
import torch
from torch.cuda.amp import autocast

from .rse import ResidualShuffleExchangeNetwork, CustomLayerNorm, gelu

class CustomShuffle(nn.Module):
    def __init__(self,scale):
        super().__init__()
        self.scale = scale

    def forward(self, x,):
        x = [x[:, :,i::self.scale] for i in range(self.scale)]
        x = torch.stack(x,dim=2)
        x = x.view((x.shape[0],x.shape[1]*self.scale,x.shape[-1]))
        return x

class TranscriptionModel(nn.Module):
    def __init__(self, window_size, n_hidden=192,n_blocks=2, vocabulary_size=128,oncpu=False):
        super().__init__()
        
        self.window_size = window_size
        n_maps1 = n_hidden // 2
        n_maps2 = n_hidden * 2
        self.dtype = torch.bfloat16
        if oncpu:
            self.dtype = torch.float32
        self.oncpu = oncpu

        scale1 = 2
        scale2 = 2
        scale3 = 2

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, n_maps1, kernel_size=3, bias=False, padding=1),
            CustomShuffle(scale=scale1),
            CustomLayerNorm(n_maps1*scale1),
            gelu(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(n_maps1*scale1, n_hidden, kernel_size=3, bias=False, padding=1),
            CustomShuffle(scale=scale2),
            CustomLayerNorm(n_hidden*scale2),
            gelu(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(n_hidden*scale2, n_maps2//scale3, kernel_size=3, bias=False, padding=1),
            CustomShuffle(scale=scale3),
            CustomLayerNorm(n_maps2),
            gelu(),
        )

        self.pre_linear = nn.Linear(n_maps2, n_hidden)

        self.rse = ResidualShuffleExchangeNetwork(n_hidden, n_blocks)

        self.post_linear = nn.Linear(n_hidden, vocabulary_size)
        self.down = scale1*scale2*scale3
        self.stride_labels = 128  # segment is labeled at positions with this stride
        self.n_frames = window_size // self.stride_labels - 1

        self.offset = nn.Parameter(torch.zeros(vocabulary_size))
        self.scale = nn.Parameter(torch.ones(vocabulary_size))

    def transform_output(self, x):
        return (x[:, :: self.stride_labels // self.down, :] - 4).permute(1, 0, 2)
    
    def calibrated_result(self, prediction,labels=None):
        with torch.no_grad():
            prediction = prediction[self.n_frames//2]
        prediction = prediction * self.scale + self.offset
        corrected_result = torch.sigmoid(prediction)
        if labels!=None:
            loss = nn.BCEWithLogitsLoss()(prediction, labels[self.n_frames//2])
            return corrected_result, loss
        else:
            return corrected_result

    def forward(self, x, targets=None, loss_fn=None,smooth=None):
        x = x*3
        x = torch.fft.rfft(x, dim=1)[:,:(self.window_size//2)].view(dtype=torch.float32)
        x = x.reshape(x.shape[0],self.window_size).to(dtype=self.dtype)
        x = x.unsqueeze(-2)
        with autocast(enabled=(not self.oncpu),dtype=self.dtype):
            if self.training:  # self.training
                x = x + torch.normal(mean=0.0, std=0.0007, size=x.shape).to(
                    x.device
                )  # to help layernorm with zero inputs
            x = self.conv1(x)  # B x C x L
            x = self.conv2(x)
            x = self.conv3(x)
            x = x.permute(0, 2, 1)  # B x L x C
            x = self.pre_linear(x) * 0.25
            x = self.rse(x)
        x = self.post_linear(x)

        trans_pred = self.transform_output(x)
        if targets!=None:
            trans_labels = targets[:, :: self.stride_labels, :].permute(1, 0, 2)
            trans_l_smooth = (1 - smooth*2) * trans_labels + smooth
            loss_lateral = loss_fn(trans_pred,trans_l_smooth)
            loss_mid = loss_fn(trans_pred[self.n_frames//2],trans_l_smooth[self.n_frames//2])
            pred_others = x[:, 0:, :] - 4
            loss_others = loss_fn(pred_others,torch.zeros_like(pred_others)+(smooth+0.1)/2)

            lateral_coef = 2 * 1/self.n_frames
            cost = loss_mid.mean() + loss_lateral.mean()*lateral_coef + loss_others.mean() * 0.01
            result, corrected_loss = self.calibrated_result(trans_pred,trans_labels)
            cost += corrected_loss * 0.1
            return cost,result
        else:
            return self.calibrated_result(trans_pred)

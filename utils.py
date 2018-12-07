import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nn_F
import glob
import os
import re


def fit_tfb(img):
    # img supoose to have shape of [3,h,w]
    return img[[2,1,0], :, :]

def save_model(model_dir, iter, model_encoder, model_decoder, inter_size = None):
    torch.save(model_encoder.state_dict(), os.path.join(
        model_dir, 'encoder_%08d.ckpt'%(iter)))
    torch.save(model_decoder.state_dict(), os.path.join(
        model_dir, 'decoder_%08d.ckpt'%(iter)))
    if not inter_size is None:
        encoder_ = glob.glob(os.path.join(model_dir, 'encoder*'))
        encoder_ = sorted(encoder_)
        decoder_ = glob.glob(os.path.join(model_dir, 'decoder*'))
        decoder_ = sorted(decoder_)
        assert (len(encoder_) == len(decoder_))
        remove_count = len(encoder_) - inter_size
        for index in range(remove_count):
            os.remove(encoder_[index])
            os.remove(decoder_[index])
        print ('remove some saved models once ')

def load_model(model_dir, model_encoder, model_decoder):
    encoder_found = glob.glob(os.path.join(model_dir, 'encoder*'))
    decoder_found = glob.glob(os.path.join(model_dir, 'decoder*'))
    encoder_ = sorted(encoder_found)[-1]
    decoder_ = sorted(decoder_found)[-1]
    iter_old = re.findall('\d+', encoder_)[0]
    assert (iter_old == re.findall('\d+', decoder_)[0])
    return model_encoder, model_decoder, int(iter_old)

def get_weighted(y):
    data_ori = y.cpu().data.numpy()
    batch_size = data_ori.shape[0]
    data = data_ori.reshape([batch_size, -1])
    Nf = 1 / np.sum(data == 1, axis = 1)
    Nb = 1 / np.sum(data == 0, axis = 1)
    weight = np.ones_like(data_ori)

    for index in range(batch_size):
        tmp_index = data_ori[index] == 1
        weight[index][tmp_index] = Nf[index]
        tmp_index = data_ori[index] == 0
        weight[index][tmp_index] = Nb[index]
    return torch.from_numpy(weight)


class seg_loss(nn.Module):
    def __init__(self,device, loss_weight = 1.0):
        super(seg_loss, self).__init__()
        self.loss_weight = loss_weight
        self.device = device

    def forward(self, x, y):
        weight = get_weighted(y) * self.loss_weight
        weight = weight.to(self.device)
        loss = nn.BCELoss(weight = weight, size_average = False)
        return loss(x,y)

    def reset_loss_weight(self, loss_weight):
        self.loss_weight = loss_weight

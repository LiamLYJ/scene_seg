import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nn_F
import glob
import os
import re

def normal_masks(input):
    # input shape: b*1 *h*w
    b,_,h,w = input.size()
    input = input.view(b, -1)
    b_min = torch.min(input, 1)[0]
    b_max = torch.max(input, 1)[0]
    input = torch.transpose(input, 0, 1)
    input = (input - b_min) / (b_max - b_min)
    input = torch.transpose(input, 0, 1)
    return input.view(b, -1, h, w)

def remap2normal(img, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    # img is b*3*h*w
    batch_size = img.shape[0]
    batch_mean = torch.Tensor(mean).repeat(batch_size).view(-1, 3, 1, 1).type(torch.DoubleTensor)
    batch_std = torch.Tensor(std).repeat(batch_size).view(-1, 3, 1, 1).type(torch.DoubleTensor)
    return img * batch_std + batch_mean

def fit_tfb(img):
    # img supoose to have shape of [3,h,w]
    return img[[2,1,0], :, :]

# load / save model of encoder and decoder
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
    model_encoder.load_state_dict(torch.load(encoder_))
    model_decoder.load_state_dict(torch.load(decoder_))
    return model_encoder, model_decoder, int(iter_old)


# load / save model of prefix
def save_model_prefix(model_dir, iter, model, prefix = 'fcn', inter_size = None):
    torch.save(model.state_dict(), os.path.join(
        model_dir, '%s_%08d.ckpt'%(prefix, iter)))
    if not inter_size is None:
        model_ = glob.glob(os.path.join(model_dir, prefix+'*'))
        model_ = sorted(model_)
        remove_count = len(model_) - inter_size
        for index in range(remove_count):
            os.remove(model_[index])
        print ('remove some saved models once ')

def load_model_prefix(model_dir, model, prefix = 'fcn'):
    model_found = glob.glob(os.path.join(model_dir, prefix+'*'))
    model_ = sorted(model_found)[-1]
    iter_old = re.findall('\d+', model_)[0]
    try:
        model.load_state_dict(torch.load(model_))
    except:
        model.load_state_dict(torch.load(model_,map_location=lambda storage, loc: storage))
    return model, int(iter_old)

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

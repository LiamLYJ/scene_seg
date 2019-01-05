import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torch.nn.functional as nn_F
from lite_net import *
import numpy as np
import argparse
from data_loader import texture_seg_dataset, get_data_direct
from utils import load_model_prefix, remap2normal, normal_masks
import os
import cv2

def main(args):
    batch_size = args.batch_size
    model_dir = args.model_dir
    save_dir = args.save_dir
    filt_stride = args.filt_stride
    filt_size = args.filt_size

    if not args.mode is None:
        device = torch.device(args.mode)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    data_set = texture_seg_dataset(args.image_dir,
                                img_size = args.img_size,
                                segmentation_regions = args.segmentation_regions,
                                texture_size = args.texture_size,
                                use_same_from = args.use_same_from)
    imgs, textures, masks = data_set.feed(batch_size)

    model_encoder = encoder_net().to(device)
    model_decoder = decoder_net().to(device)
    filt_adp = nn.AdaptiveAvgPool2d((filt_size,filt_size))

    model_encoder, model_decoder, iter_old = load_model(model_dir, model_encoder, model_decoder)
    print ('load model from %d iter'%(iter_old))

    imgs = torch.from_numpy(imgs)
    textures = torch.from_numpy(textures)
    masks = torch.from_numpy(masks)

    imgs = imgs.type(torch.FloatTensor).to(device)
    textures = textures.type(torch.FloatTensor).to(device)
    masks = masks.type(torch.FloatTensor).to(device)

    encoder_img, vgg_features = model_encoder(imgs)
    encoder_texture, _ = model_encoder(textures)

    filt = filt_adp(encoder_texture).to(device)

    correlations = []
    for index in range(batch_size):
        t0 = encoder_img[index].cuda()
        t1 = filt[index].cuda()
        padding = (filt_stride - 1) * t0.shape[-1] - filt_stride + filt.shape[-1]
        padding = int(padding / 2)
        correlations.append(nn_F.conv2d(t0.unsqueeze(0), t1.unsqueeze(0), stride = filt_stride, padding = padding))
    correlations = torch.cat(correlations, 0)
    output_masks, _ = model_decoder(correlations, vgg_features)
    print ('output_masks: ', output_masks.shape)
    print ('img shape: ', imgs.shape)
    print ('masks shape:', masks.shape)

    imgs = remap2normal(imgs.cpu())
    textures = remap2normal(textures.cpu())
    output_masks = normal_masks(output_masks.cpu())

    for index in range(batch_size):
        torchvision.utils.save_image(imgs[index], os.path.join(save_dir, 'input_img_%02d.png'%(index)))
        torchvision.utils.save_image(masks[index], os.path.join(save_dir, 'gt_%02d.png'%(index)))
        torchvision.utils.save_image(output_masks[index], os.path.join(save_dir, 'output_%02d.png'%(index)))
        torchvision.utils.save_image(textures[index], os.path.join(save_dir, 'texture_%02d.png'%(index)))


def load_direct(args):
    model_dir = args.model_dir
    save_dir = args.save_dir
    batch_size = args.batch_size
    n_class = args.segmentation_regions

    if not args.mode is None:
        device = torch.device(args.mode)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    net_model = lite_net().to(device)
    try:
        net_model, iter_old = load_model_prefix(model_dir, net_model, prefix = 'concat')
        print ('load model from %d iteration'%(iter_old))
    except:
        raise ValueError('something wrong when loading pre-trained model')

    all_imgs, all_textures = get_data_direct(img_size = args.img_size, imgs_dir = args.imgs_dir,
                                                texture_size = args.img_size, textures_dir = args.textures_dir)
    all_batch_size = all_imgs.shape[0]
    print ('all batch_size is:', all_batch_size)
    iter_num = all_batch_size // batch_size

    # only allow to use remain is 0!!!
    assert (all_batch_size % batch_size == 0)

    for iter in range(iter_num):
        imgs = all_imgs[iter*batch_size:(iter+1)*batch_size, ...]
        textures = all_textures[iter*batch_size:(iter+1)*batch_size, ...]
        imgs = torch.from_numpy(imgs)
        textures = torch.from_numpy(textures)
        input = torch.cat([imgs, textures], dim = 1)
        input = input.type(torch.FloatTensor).to(device)
        output_masks = net_model(input).to(device)

        print ('output_masks: ', output_masks.shape)
        print ('img shape: ', imgs.shape)

        imgs = remap2normal(imgs.cpu())
        output_masks = normal_masks(output_masks.cpu())

        for index in range(batch_size):
            torchvision.utils.save_image(imgs[index], os.path.join(save_dir, 'input_img_%d_%02d.png'%(iter, index)))
            torchvision.utils.save_image(output_masks[index,...], os.path.join(save_dir, 'output_%d_%02d.png'%(iter,index)))


if __name__ == '__main__':
    # path
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./models/lite_scene_model' , help='path for saving trained models')
    parser.add_argument('--image_dir', type=str, default='./dataset/dtd/images', help='directory for images from')
    parser.add_argument('--mode', type=str, default=None, help = 'mode to use ')
    parser.add_argument('--use_same_from', type=bool, default=True, help = 'if use the same texture from that same')
    parser.add_argument('--save_dir', type=str, default='./tmp_test', help='directory for saving ')

    # Model parameters
    parser.add_argument('--img_size', type=int , default=256, help='input image size')
    parser.add_argument('--segmentation_regions', type=int , default=4, help='number of segmentation_regions')

    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--imgs_dir', type=str, default='./imgs_dir', help='directory for images from')
    parser.add_argument('--textures_dir', type=str, default='./textures_dir', help='directory for images from')

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # main(args)
    load_direct(args)

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torch.nn.functional as nn_F
from nets import encoder_net, decoder_net
import numpy as np
import argparse
from data_loader import texture_seg_dataset
from utils import seg_loss, load_model
import os
import cv2

def main(args):
    batch_size = args.batch_size
    model_dir = args.model_dir
    save_dir = args.save_dir

    if not args.mode is None:
        device = torch.device(args.mode)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    data_set = texture_seg_dataset(args.image_dir,
                                img_size = args.img_size,
                                segmentation_regions = args.segmentation_regions,
                                texture_size = args.texture_size,)
    imgs, textures, masks = data_set.feed(batch_size)

    model_encoder = encoder_net().to(device)
    model_decoder = decoder_net().to(device)

    model_encoder, model_decoder, iter_old = load_model(model_dir, model_encoder, model_decoder)
    print ('load model from %d iter'%(iter_old))

    filt_adp = nn.AdaptiveAvgPool2d((5,5))


    imgs = torch.from_numpy(imgs)
    textures = torch.from_numpy(textures)
    masks = torch.from_numpy(masks)

    imgs = imgs.type(torch.FloatTensor).to(device)
    textures = textures.type(torch.FloatTensor).to(device)
    masks = masks.type(torch.FloatTensor).to(device)

    encoder_img, vgg_features = model_encoder(imgs)
    encoder_texture, _ = model_encoder(textures)

    filt = filt_adp(encoder_texture).to(device)
    # correlations = nn_F.conv2d(encoder_img, filt, stride = 1, padding = 2)
    correlations = []
    for index in range(batch_size):
        t0 = encoder_img[index].cuda()
        t1 = filt[index].cuda()
        correlations.append(nn_F.conv2d(t0.unsqueeze(0), t1.unsqueeze(0), stride = 1, padding = 2))
    correlations = torch.cat(correlations, 0)
    output_masks, _ = model_decoder(correlations, vgg_features)
    print ('output_masks: ', output_masks.shape)
    print ('img shape: ', imgs.shape)
    print ('masks shape:', masks.shape)

    torchvision.utils.save_image(imgs/ 255.0, os.path.join(save_dir, 'input_img.png'))
    torchvision.utils.save_image(masks, os.path.join(save_dir, 'gt.png'))
    torchvision.utils.save_image(output_masks, os.path.join(save_dir, 'output.png'))
    torchvision.utils.save_image(textures / 255.0, os.path.join(save_dir, 'texture.png'))

if __name__ == '__main__':
    # path
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./models/' , help='path for saving trained models')
    parser.add_argument('--image_dir', type=str, default='./images', help='directory for images from')
    parser.add_argument('--mode', type=str, default=None, help = 'mode to use ')
    parser.add_argument('--save_dir', type=str, default='./save_results', help='directory for saving ')

    # Model parameters
    parser.add_argument('--img_size', type=int , default=256, help='input image size')
    parser.add_argument('--segmentation_regions', type=int , default=3, help='number of segmentation_regions')
    parser.add_argument('--texture_size', type=int , default=64, help='texture input size')

    parser.add_argument('--batch_size', type=int, default=2)

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    main(args)

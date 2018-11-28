import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as nn_F
from nets import encoder_net, decoder_net
import numpy as np
import argparse
from data_loader import texture_seg_dataset


def main(args):
    num_iter = args.num_iter
    batch_size = args.batch_size
    data_set = texture_seg_dataset(args.image_dir,
                                img_size = args.img_size,
                                segmentation_regions = args.segmentation_regions,
                                texture_size = args.texture_size,)


    model_encoder = encoder_net()
    model_decoder = decoder_net()

    filt_adp = nn.AdaptiveAvgPool2d((5,5))

    for iter in range(num_iter):

        imgs, textures, masks = data_set.feed(batch_size)
        imgs = torch.from_numpy(imgs)
        textures = torch.from_numpy(textures)
        masks = torch.from_numpy(masks)
        imgs = imgs.type(torch.FloatTensor)
        textures = textures.type(torch.FloatTensor)
        masks = masks.type(torch.FloatTensor)

        encoder_img, vgg_features = model_encoder(imgs)
        encoder_texture, _ = model_encoder(textures)

        filt = filt_adp(encoder_texture)
        # correlations = nn_F.conv2d(encoder_img, filt, stride = 1, padding = 2)
        correlations = []
        for index in range(batch_size):
            t0 = encoder_img[index]
            t1 = filt[index]
            correlations.append(nn_F.conv2d(t0.unsqueeze(0), t1.unsqueeze(0), stride = 1, padding = 2))
        correlations = torch.cat(correlations, 0)
        output_masks, _ = model_decoder(correlations, vgg_features)


if __name__ == '__main__':
    # path
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--image_dir', type=str, default='./images', help='directory for images from')
    parser.add_argument('--log_step', type=int , default=1, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')

    # Model parameters
    parser.add_argument('--img_size', type=int , default=256, help='input image size')
    parser.add_argument('--segmentation_regions', type=int , default=3, help='number of segmentation_regions')
    parser.add_argument('--texture_size', type=int , default=64, help='texture input size')

    parser.add_argument('--num_iter', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    main(args)

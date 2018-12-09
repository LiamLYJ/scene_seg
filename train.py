import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as nn_F
from nets import encoder_net, decoder_net
import numpy as np
import argparse
from data_loader import texture_seg_dataset
from utils import seg_loss, load_model, save_model, fit_tfb
import os
from tensorboardX import SummaryWriter


def main(args):
    total_iter = args.total_iter
    batch_size = args.batch_size
    loss_weight = args.loss_weight
    model_dir = args.model_dir
    save_size = args.save_inter_size
    filt_stride = args.filt_stride
    filt_size = args.filt_size

    if not args.mode is None:
        device = torch.device(args.mode)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    writer = SummaryWriter(args.log_dir)

    data_set = texture_seg_dataset(args.image_dir,
                                img_size = args.img_size,
                                segmentation_regions = args.segmentation_regions,
                                texture_size = args.texture_size,
                                use_same_from = args.use_same_from)

    model_encoder = encoder_net().to(device)
    model_decoder = decoder_net().to(device)
    filt_adp = nn.AdaptiveAvgPool2d((filt_size,filt_size))

    # load model
    iter_old = 0
    try:
        model_encoder, model_decoder, iter_old = load_model(model_dir, model_encoder, model_decoder)
        print ('load model from %d iteration'%(iter_old))
    except:
        print ('start from zero training')

    # loss and optimizer
    params = list(model_encoder.parameters()) + list(model_decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    if args.loss_type == 'seg_loss':
        criterion = seg_loss(device, loss_weight)
    else:
        criterion = nn.MSELoss()

    for iter in range(iter_old + 1, total_iter):

        imgs, textures, masks = data_set.feed(batch_size)
        imgs = torch.from_numpy(imgs)
        textures = torch.from_numpy(textures)
        masks = torch.from_numpy(masks)

        imgs = imgs.type(torch.FloatTensor).to(device)
        textures = textures.type(torch.FloatTensor).to(device)
        masks = masks.type(torch.FloatTensor).to(device)

        encoder_img, vgg_features = model_encoder(imgs)
        encoder_texture, _ = model_encoder(textures)

        # print ('endoder_img :', encoder_img.shape)
        # print ('encoder_texture: ', encoder_texture.shape)
        filt = filt_adp(encoder_texture).to(device)
        # filt = encoder_texture.to(device)
        # print ('filt :', filt.shape)

        # correlations = nn_F.conv2d(encoder_img, filt, stride = 1, padding = 2)
        correlations = []
        for index in range(batch_size):
            t0 = encoder_img[index].cuda()
            t1 = filt[index].cuda()
            padding = (filt_stride - 1) * t0.shape[-1] - filt_stride + filt.shape[-1]
            padding = int(padding / 2)
            # print ('padding: ', padding)
            correlations.append(nn_F.conv2d(t0.unsqueeze(0), t1.unsqueeze(0), stride = filt_stride, padding = padding))
        correlations = torch.cat(correlations, 0)
        output_masks, _ = model_decoder(correlations, vgg_features)
        # print ('output_masks: ', output_masks.shape)

        loss = criterion(output_masks, masks)

        model_decoder.zero_grad()
        model_encoder.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % args.log_step == 0:
            print('Iter [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
              .format(iter, total_iter, loss.item(), np.exp(loss.item())))
            writer.add_scalar('loss',loss.item(), iter)
            # just choose one
            writer.add_image('input', fit_tfb(imgs[0]), iter)
            writer.add_image('texture', fit_tfb(textures[0]), iter)
            writer.add_image('mask', masks[0], iter)
            writer.add_image('output', output_masks[0], iter)

        # Save the model checkpoints
        if (iter+1) % args.save_step == 0:
            save_model(model_dir, iter, model_encoder, model_decoder, inter_size = save_size)
            print ('model saved once : total_inter: %d, iteration : %d'%(total_iter, iter))

    writer.close()

if __name__ == '__main__':
    # path
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./models/' , help='path for saving trained models')
    parser.add_argument('--image_dir', type=str, default='./dataset/scene_images', help='directory for images from')
    parser.add_argument('--log_dir', type=str, default='./logs/' , help='path for saving tensorboard')
    parser.add_argument('--log_step', type=int , default=2, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    parser.add_argument('--save_inter_size', type=int , default=2, help='how many to keep for saving')
    parser.add_argument('--mode', type=str, default=None, help = 'mode to use ')
    parser.add_argument('--use_same_from', type=bool, default=True, help = 'if use the same texture from that same')

    # Model parameters
    parser.add_argument('--img_size', type=int , default=256, help='input image size')
    parser.add_argument('--segmentation_regions', type=int , default=3, help='number of segmentation_regions')
    parser.add_argument('--texture_size', type=int , default=64, help='texture input size')
    parser.add_argument('--filt_stride', type=int , default=1, help='convolution stride of textural filt')
    parser.add_argument('--filt_size', type=int , default=11, help='convolution filt size of textural filt')

    parser.add_argument('--total_iter', type=int, default=9999999)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--loss_weight', type=float, default=1.0)
    parser.add_argument('--loss_type', type=str, default= 'seg_loss', help='seg_loss or rms_loss')

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    main(args)

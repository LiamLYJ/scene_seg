import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as nn_F
from fcn_net import *
import numpy as np
import argparse
from data_loader import fcn_dataset
from utils import load_fcn_model, save_fcn_model, fit_tfb, seg_loss
import os
from tensorboardX import SummaryWriter


def main(args):
    total_iter = args.total_iter
    batch_size = args.batch_size
    loss_weight = args.loss_weight
    model_dir = args.model_dir
    save_size = args.save_inter_size
    n_class = args.segmentation_regions

    if not args.mode is None:
        device = torch.device(args.mode)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    writer = SummaryWriter(args.log_dir)

    data_set = fcn_dataset('./dataset/scene_images', img_size = 256, segmentation_regions= 4,)

    vgg_model = VGGNet(requires_grad=True).to(device)
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class).to(device)

    # load model
    iter_old = 0
    try:
        fcn_model, iter_old = load_fcn_model(model_dir, fcn_model)
        print ('load model from %d iteration'%(iter_old))
    except:
        print ('start from zero training')

    # loss and optimizer
    params = list(fcn_model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    if args.loss_type == 'seg_loss':
        criterion = seg_loss(device, loss_weight)
    else:
        criterion = nn.MSELoss()

    for iter in range(iter_old + 1, total_iter):

        imgs, masks = data_set.feed(batch_size)
        imgs = torch.from_numpy(imgs)
        masks = torch.from_numpy(masks)

        imgs = imgs.type(torch.FloatTensor).to(device)
        masks = masks.type(torch.FloatTensor).to(device)

        output_masks = fcn_model(imgs).to(device)

        loss = criterion(output_masks, masks)

        fcn_model.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % args.log_step == 0:
            print('Iter [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
              .format(iter, total_iter, loss.item(), np.exp(loss.item())))
            writer.add_scalar('loss',loss.item(), iter)
            # just choose one from batch
            writer.add_image('input', fit_tfb(imgs[0]), iter)
            for index_class in range(n_class):
                writer.add_image('mask_%d'%(index_class), masks[0][index_class], iter)
                writer.add_image('output_%d'%(index_class), output_masks[0][index_class], iter)

        # Save the model checkpoints
        if (iter+1) % args.save_step == 0:
            save_fcn_model(model_dir, iter, fcn_model, inter_size = save_size)
            print ('model saved once : total_inter: %d, iteration : %d'%(total_iter, iter))

    writer.close()

if __name__ == '__main__':
    # path
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./models/fcn_scene_model' , help='path for saving trained models')
    parser.add_argument('--image_dir', type=str, default='./dataset/scene_images', help='directory for images from')
    parser.add_argument('--log_dir', type=str, default='./logs/fcn_scene/' , help='path for saving tensorboard')
    parser.add_argument('--log_step', type=int , default=2, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    parser.add_argument('--save_inter_size', type=int , default=2, help='how many to keep for saving')
    parser.add_argument('--mode', type=str, default=None, help = 'mode to use ')

    # Model parameters
    parser.add_argument('--img_size', type=int , default=256, help='input image size')
    parser.add_argument('--segmentation_regions', type=int , default=4, help='number of segmentation_regions')

    parser.add_argument('--total_iter', type=int, default=9999999)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--loss_weight', type=float, default=1.0)
    parser.add_argument('--loss_type', type=str, default= 'seg_loss', help='seg_loss or rms_loss')

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    main(args)

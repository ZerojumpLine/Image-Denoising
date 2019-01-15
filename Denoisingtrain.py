import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import data_generator as dg
from data_generator import DenoisingDataset
import logging
import random
import torch.nn.functional as F
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
import common
from modeledsr import EDSR


# Params
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model', default='EDSR', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--set_dir', default='/data/Val/', type=str, help='directory of test dataset')

parser.add_argument('--set_names', default=['Noisy'], help='directory of test dataset')
parser.add_argument('--set_namesori', default='Clean', help='directory of test dataset')

parser.add_argument('--train_data', default='/data/Clean', type=str, help='path of train data')
parser.add_argument('--train_data_noise', default='/data/Noisy', type=str, help='path of train data')

parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=10, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-6, type=float, help='initial learning rate for Adam') # 1e-4
args = parser.parse_args()

batch_size = args.batch_size
cuda = torch.cuda.is_available()
n_epoch = args.epoch
sigma = args.sigma

exname = 'p64s20b64ISO33ori'

save_dir = os.path.join('models', args.model+'_' + exname)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        # return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)
        return torch.nn.functional.l1_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def log(*args, **kwargs):
     print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)

def batch_PSNR(img, imclean):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:])
    return (PSNR/Img.shape[0])


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='Train' + args.model +  exname + '.log',
                        filemode='a')

    # model selection
    print('===> Building model')
    model = EDSR()
    
    initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        # maybe I can modify this
        print('resuming by loading epoch %03d' % initial_epoch)
        model = torch.load(os.path.join(save_dir, 'model_003_009.pth'))
        initial_epoch = 7
    model.train()
    criterion = sum_squared_error()
    if cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates

    for epoch in range(initial_epoch, n_epoch):

        for subepoch in range(0,10):

            logging.info('epoch' + str(epoch))
            scheduler.step(epoch)  # step to the learning rate in this epcoh
            xs, xn = dg.datagenerator(data_dir=args.train_data, data_dir_noise=args.train_data_noise, batch_size=batch_size)
            listr = list(range(0,xs.shape[0]))
            random.shuffle(listr)
            xs = xs[listr, :, :, :]
            xn = xn[listr, :, :, :]
            xs = xs.astype('float32')/255.0
            xn = xn.astype('float32')/255.0
            xs = torch.from_numpy(xs.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
            xn = torch.from_numpy(xn.transpose((0, 3, 1, 2)))  # tensor of the clean patches, NXCXHXW
            DDataset = DenoisingDataset(xs, xn)
            DLoader = DataLoader(dataset=DDataset, num_workers=8, drop_last=True, batch_size=batch_size, shuffle=True)
            epoch_loss = 0
            start_time = time.time()

            for n_count, batch_yx in enumerate(DLoader):
                    optimizer.zero_grad()
                    if cuda:
                        batch_x, batch_y = batch_yx[1].cuda(), batch_yx[0].cuda()
                    loss = criterion(model(batch_y), batch_x)
                    epoch_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    if n_count % 100 == 0:
                        print('%4d %4d / %4d loss = %2.4f' % (epoch+1, n_count, xs.size(0)//batch_size, loss.item()/batch_size))
            elapsed_time = time.time() - start_time

            log('epoch = %4d , subspoch = %4d, loss = %4.4f , time = %4.2f s' % (epoch+1, subepoch+1, epoch_loss/n_count, elapsed_time))
            # np.savetxt('train_result.txt', np.hstack((epoch+1, epoch_loss/n_count, elapsed_time)), fmt='%2.4f')
            logging.info('epoch' + str(epoch+1) + ' subepoch ' + str(subepoch+1) + '  loss ' + str(epoch_loss/n_count) + '  time ' + str(elapsed_time))

            model.train()
            for set_cur in args.set_names:

                psnrs = []
                ssims = []

                for im in os.listdir(os.path.join(args.set_dir, set_cur)):
                    if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                        xori = np.array(imread(os.path.join(args.set_dir, args.set_namesori, im)),
                                        dtype=np.float32) / 255.0
                        x = np.array(imread(os.path.join(args.set_dir, set_cur, im)), dtype=np.float32) / 255.0
                        y = x
                        y = y.astype(np.float32)
                        yt = y.transpose((2, 0, 1))
                        y_ = torch.from_numpy(yt).view(1, -1, y.shape[0], y.shape[1])

                        torch.cuda.synchronize()
                        y_ = y_.cuda()

                        # split into 16 128*128#########################################################################
                        ## cumbsome!!! for RCAN or when memory is not enough

                        x_ = np.zeros((3,1024,1024))
                        for kx in range(0,8):
                            for ky in range(0,8):
                                kxstart = kx * 128
                                kystart = ky * 128
                                # print(ky)
                                y_1 = y_[:, :, kxstart: kxstart + 128, kystart: kystart + 128]
                                x_1 = model(y_1)
                                x1 = x_1.view(3, 128, 128)
                                x1 = x1.cpu()
                                x1 = x1.detach().numpy().astype(np.float32)
                                x_[:, kxstart: kxstart + 128, kystart: kystart + 128] = x1

                        ################################################################################################

                        # x_ = model(y_)  # inference
                        # x_ = x_.view(3, y.shape[0], y.shape[1])
                        # x_ = x_.cpu()
                        # x_ = x_.detach().numpy().astype(np.float32)

                        x_ = x_.transpose((1, 2, 0))
                        torch.cuda.synchronize()
                        print('%10s : %10s ' % (set_cur, im))

                        psnr_x_ = compare_psnr(xori, x_)
                        ssim_x_ = compare_ssim(xori, x_, multichannel=True)
                        psnrs.append(psnr_x_)
                        ssims.append(ssim_x_)
                psnr_avg = np.mean(psnrs)
                ssim_avg = np.mean(ssims)
                psnrs.append(psnr_avg)
                ssims.append(ssim_avg)
                log('Datset: {0:10s} \n  PSNR = {1:2.2f}dB, SSIM = {2:1.4f}'.format(set_cur, psnr_avg, ssim_avg))
                logging.info('epoch' + str(epoch + 1) + ' subepoch ' + str(subepoch+1) + ' val PSNR = ' + str(psnr_avg) + ' val SSIM = ' + str(ssim_avg))
            torch.save(model, os.path.join(save_dir, 'model_%03d_%03d.pth' % (epoch + 1, subepoch + 1)))
        torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))










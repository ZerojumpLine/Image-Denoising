
# run this to test the model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import time, datetime
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
import torch.nn.functional as F
import warnings
from modeledsr import EDSR

warnings.filterwarnings("ignore")


def save_result(result, path):
    path = path if path.find('.') != -1 else path+'.png'
    ext = os.path.splitext(path)[-1]
    if ext in ('.txt', '.dlm'):
        np.savetxt(path, result, fmt='%2.4f')
    else:
        imsave(path, np.clip(result, 0, 1))

if __name__ == '__main__':

    ISOsetall = ['11','12','21','22','31','32','33']
    # the images are divided into 7 classes based on ISO, 7 models are trained separately
    # the ranges are
    # 11: IS0 [0,500]
    # 12: IS0 [501,1000]
    # 21: IS0 [1001,1500]
    # 22: IS0 [1501,2000]
    # 31: IS0 [2001,2500]
    # 32: IS0 [2501,3000]
    # 33: IS0 [3001,3200]


    for ISOset in ISOsetall:

        # result_dir = './testresults'
        result_dir = './evalresults'
        # set_dir = '/vol/medic02/users/zl9518/Denoise/data/Testing_ISO/Testing_re' + str(ISOset) + '/Noisy'
        set_dir = '/data/Denoise/data/Eval_ISO/Eval_re' + str(ISOset) + '/Noisy'
        modelpath = '/data/Denoise/DnCNN_pytorch/models/exp6models/ISO' + str(ISOset) + '/model.pth'

        model = torch.load(modelpath)

        print('model loaded ' + str(ISOset))


        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        for im in os.listdir(os.path.join(set_dir)):
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):



                x = np.array(imread(os.path.join(set_dir, im)), dtype=np.float32) / 255.0
                y = x
                y = y.astype(np.float32)

                # self-ensemble##########################################################################
                start_time = time.time()
                y1 = np.flipud(y)
                yt1 = y1.transpose((2, 0, 1))
                y_1 = torch.from_numpy(yt1.copy()).view(1, -1, y1.shape[0], y1.shape[1])
                torch.cuda.synchronize()
                y_1 = y_1.cuda()
                x_1 = model(y_1)
                x_1 = x_1.view(3, y.shape[0], y.shape[1])
                x_1 = x_1.cpu()
                x_1 = x_1.detach().numpy().astype(np.float32)
                x_1 = x_1.transpose((1, 2, 0))
                x_1 = np.flipud(x_1)

                y2 = np.rot90(y)
                yt2 = y2.transpose((2, 0, 1))
                y_2 = torch.from_numpy(yt2.copy()).view(1, -1, y.shape[0], y.shape[1])
                torch.cuda.synchronize()
                y_2 = y_2.cuda()
                x_2 = model(y_2)
                x_2 = x_2.view(3, y.shape[0], y.shape[1])
                x_2 = x_2.cpu()
                x_2 = x_2.detach().numpy().astype(np.float32)
                x_2 = x_2.transpose((1, 2, 0))
                x_2 = np.rot90(x_2, k=3)

                y3 = np.flipud(np.rot90(y))
                yt3 = y3.transpose((2, 0, 1))
                y_3 = torch.from_numpy(yt3.copy()).view(1, -1, y.shape[0], y.shape[1])
                torch.cuda.synchronize()
                y_3 = y_3.cuda()
                x_3 = model(y_3)
                x_3 = x_3.view(3, y.shape[0], y.shape[1])
                x_3 = x_3.cpu()
                x_3 = x_3.detach().numpy().astype(np.float32)
                x_3 = x_3.transpose((1, 2, 0))
                x_3 = np.rot90(np.flipud(x_3), k=3)

                y4 = np.rot90(y, k=2)
                yt4 = y4.transpose((2, 0, 1))
                y_4 = torch.from_numpy(yt4.copy()).view(1, -1, y.shape[0], y.shape[1])
                torch.cuda.synchronize()
                y_4 = y_4.cuda()
                x_4 = model(y_4)
                x_4 = x_4.view(3, y.shape[0], y.shape[1])
                x_4 = x_4.cpu()
                x_4 = x_4.detach().numpy().astype(np.float32)
                x_4 = x_4.transpose((1, 2, 0))
                x_4 = np.rot90(x_4, k=2)

                y5 = np.flipud(np.rot90(y, k=2))
                yt5 = y5.transpose((2, 0, 1))
                y_5 = torch.from_numpy(yt5.copy()).view(1, -1, y.shape[0], y.shape[1])
                torch.cuda.synchronize()
                y_5 = y_5.cuda()
                x_5 = model(y_5)
                x_5 = x_5.view(3, y.shape[0], y.shape[1])
                x_5 = x_5.cpu()
                x_5 = x_5.detach().numpy().astype(np.float32)
                x_5 = x_5.transpose((1, 2, 0))
                x_5 = np.rot90(np.flipud(x_5), k=2)

                y6 = np.rot90(y, k=3)
                yt6 = y6.transpose((2, 0, 1))
                y_6 = torch.from_numpy(yt6.copy()).view(1, -1, y.shape[0], y.shape[1])
                torch.cuda.synchronize()
                y_6 = y_6.cuda()
                x_6 = model(y_6)
                x_6 = x_6.view(3, y.shape[0], y.shape[1])
                x_6 = x_6.cpu()
                x_6 = x_6.detach().numpy().astype(np.float32)
                x_6 = x_6.transpose((1, 2, 0))
                x_6 = np.rot90(x_6)

                y7 = np.flipud(np.rot90(y, k=3))
                yt7 = y7.transpose((2, 0, 1))
                y_7 = torch.from_numpy(yt7.copy()).view(1, -1, y.shape[0], y.shape[1])
                torch.cuda.synchronize()
                y_7 = y_7.cuda()
                x_7 = model(y_7)
                x_7 = x_7.view(3, y.shape[0], y.shape[1])
                x_7 = x_7.cpu()
                x_7 = x_7.detach().numpy().astype(np.float32)
                x_7 = x_7.transpose((1, 2, 0))
                x_7 = np.rot90(np.flipud(x_7))

                y8 = y
                yt8 = y8.transpose((2, 0, 1))
                y_8 = torch.from_numpy(yt8).view(1, -1, y.shape[0], y.shape[1])
                torch.cuda.synchronize()
                y_8 = y_8.cuda()
                x_8 = model(y_8)
                x_8 = x_8.view(3, y.shape[0], y.shape[1])
                x_8 = x_8.cpu()
                x_8 = x_8.detach().numpy().astype(np.float32)
                x_8 = x_8.transpose((1, 2, 0))
                x_8 = x_8

                x_ = (x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8) / 8

                ##############################################################################################

                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                print('%10s : %2.4f second' % (im, elapsed_time))

                name, ext = os.path.splitext(im)

                save_result(x_, path=os.path.join(result_dir, name + ext))










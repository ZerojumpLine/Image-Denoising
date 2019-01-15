import glob
import cv2
import numpy as np
# from multiprocessing import Pool
from torch.utils.data import Dataset
import torch
import random

patch_size, stride = 64, 20
aug_times = 1
# scales = [1, 0.9, 0.8]
scales = [1]
# batch_size = 16


class DenoisingDataset(Dataset):
    """Dataset wrapping tensors.
    Arguments:
        xs (Tensor): clean image patches
        sigma: noise level, e.g., 25
    """
    def __init__(self, xs, xn):
        super(DenoisingDataset, self).__init__()
        self.xs = xs
        self.xn = xn

    def __getitem__(self, index):
        batch_x = self.xs[index]
        # noise = torch.randn(batch_x.size()).mul_(self.sigma/255.0)
        batch_y = self.xn[index]
        return batch_y, batch_x

    def __len__(self):
        return self.xs.size(0)


def show(x, title=None, cbar=False, figsize=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    plt.imshow(x, interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def data_aug(img, mode=0):

    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def gen_patches(file_name, path_namen):

    img = cv2.imread(file_name)  # RGB  scale

    # the number '10' should be changed according to the path of training data...
    namespec = file_name[10:]
    file_namen = path_namen+namespec

    imgn = cv2.imread(file_namen)  # RGB scale
    h, w, c = img.shape
    patches = []
    patchesn = []
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        img_scaled = cv2.resize(img, (w_scaled, h_scaled), interpolation=cv2.INTER_CUBIC)
        imgn_scaled = cv2.resize(imgn, (w_scaled, h_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches

        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):

                x = img_scaled[i:i+patch_size, j:j+patch_size]
                xn = imgn_scaled[i:i+patch_size, j:j+patch_size]
                # i_index = np.random.randint(0, h_scaled - patch_size + 1)
                # j_index = np.random.randint(0, w_scaled - patch_size + 1)
                # x = img_scaled[i_index:i_index + patch_size, j_index:j_index + patch_size]
                # xn = imgn_scaled[i_index:i_index + patch_size, j_index:j_index + patch_size]
                for k in range(0, aug_times):
                    modenum = np.random.randint(0, 8)
                    # modenum = 0
                    x_aug = data_aug(x, mode=modenum)
                    xn_aug = data_aug(xn, mode=modenum)
                    patches.append(x_aug)
                    patchesn.append(xn_aug)
    return patches, patchesn


def datagenerator(data_dir, data_dir_noise, batch_size, verbose=True):
    file_list = glob.glob(data_dir+'/*.png')  # get name list of all .png files

    random.shuffle(file_list)
    file_list = file_list[0:np.minimum(len(file_list),15)]



    # initrialize
    data = []
    data_noise = []
    # generate patches
    for i in range(len(file_list)):
        patch, patchn = gen_patches(file_list[i], data_dir_noise)
        data.append(patch)
        data_noise.append(patchn)
        if verbose:
            print(str(i+1) + '/' + str(len(file_list)) + ' is done ^_^')
    data = np.array(data, dtype='uint8')
    data = data.reshape((data.shape[0]*data.shape[1], data.shape[2], data.shape[3], 3))
    discard_n = len(data)-len(data)//batch_size*batch_size
    data = np.delete(data, range(discard_n), axis=0)

    data_noise = np.array(data_noise, dtype='uint8')
    data_noise = data_noise.reshape((data_noise.shape[0] * data_noise.shape[1], data_noise.shape[2], data_noise.shape[3], 3))
    data_noise = np.delete(data_noise, range(discard_n), axis=0)
    print('^_^-training data finished-^_^')
    return data, data_noise


if __name__ == '__main__': 

    data = datagenerator(data_dir='data/Train400')



from __future__ import print_function
from torch.optim import lr_scheduler
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy import ndimage
from tqdm import tqdm
import os
import os.path as osp
import time
import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ddnet_multi_test_0910_12081_cross_chun_csc4_vgg16 import ddnet
from skimage import segmentation as seg
import glob
from losses_pytorch.dice_loss import GDiceLoss2
from loss import DC_and_CE_loss,SoftDiceLoss
from sklearn.metrics.pairwise import cosine_similarity

datadir = '/home/guolibao/cardiac-jbhi/'
voc_root = os.path.join(datadir, 'cardiac-4ch')
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()  # mean squared error
        else:
            self.loss = nn.BCELoss()  # Binary Cross Entropy

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


def read_images(root_dir, train):
    txt_fname = root_dir + '/dic/' + ('train.txt' if train else 'val.txt')
    with open(txt_fname, 'r')as f:
        images = f.read().split()

    if train:
        data_list = [os.path.join(root_dir, 'train', i) for i in images]

        label_list = [os.path.join(root_dir, 'train_labels', i) for i in images]
    else:
        data_list = [os.path.join(root_dir, 'val', i) for i in images]
        label_list = [os.path.join(root_dir, 'val_labels', i) for i in images]
    return data_list, label_list


class GTAdataset(Dataset):
    '''GTA dataset'''

    def __init__(self, root_dir=voc_root, train=True, trsf=None):
        self.root_dir = root_dir
        self.trsf = trsf
        self.data_list, self.label_list = read_images(root_dir, train)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image, label = self.data_list[idx], self.label_list[idx]
        image, label = Image.open(image).convert('RGB'), Image.open(label)
        image = transforms.Resize((256,256), interpolation=Image.BILINEAR)(image)
        label = transforms.Resize((256,256), interpolation=Image.BILINEAR)(label)
        label1 = label.convert('RGB')
        sample = {'image': image, 'label': label, 'label_rgb': label1}
        if self.trsf:
            sample = self.trsf(sample)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        image, label, label_rgb = sample['image'], sample['label'], sample['label_rgb']
        image = transforms.ToTensor()(image)
        label = torch.from_numpy(np.array(label, dtype='int'))
        label_rgb = transforms.ToTensor()(label_rgb)
        return {'image': image, 'label': label, 'label_rgb': label_rgb}


class Normalize(object):
    def __init__(self, mean=[0., 0., 0.], std=[1., 1., 1.]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label, label_rgb = sample['image'], sample['label'], sample['label_rgb']
        image = transforms.Normalize(self.mean, self.std)(image)
        label_rgb = transforms.Normalize(self.mean, self.std)(label_rgb)
        return {'image': image, 'label': label, 'label_rgb': label_rgb}


def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)


def fast_hist(label_pred, label_gt, num_category):
    mask = (label_gt >= 0) & (label_gt < num_category)  # include background
    hist = np.bincount(
        num_category * label_pred[mask] + label_gt[mask].astype(int),
        minlength=num_category ** 2).reshape(num_category, num_category)
    return hist


def evaluation_metrics(label_preds, label_gts, num_category):
    """Returns evaluation result.
      - pixel accuracy
      - mean accuracy
      - mean IoU
      - frequency weighted IoU
      - dice
      - recall
      - pre
    """
    hist = np.zeros((num_category, num_category))
    for p, g in zip(label_preds, label_gts):
        tmp = (g < 10)
        hist += fast_hist(p[tmp], g[tmp], num_category)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        macc = np.diag(hist) / hist.sum(axis=0)
    macc = np.nanmean(macc)
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(hist) / (hist.sum(axis=0) + hist.sum(axis=1) - np.diag(hist))
    # miou = np.nanmean(iou)
    with np.errstate(divide='ignore', invalid='ignore'):
        dice = 2 * np.diag(hist) / (hist.sum(axis=0) + hist.sum(axis=1))
    # mdice = np.nanmean(dice)
    with np.errstate(divide='ignore', invalid='ignore'):
        recall = np.diag(hist) / hist.sum(axis=1)
    # mrecall = np.nanmean(recall)
    with np.errstate(divide='ignore', invalid='ignore'):
        pre = np.diag(hist) / hist.sum(axis=0)
    # mpre = np.nanmean(pre)
    freq = hist.sum(axis=0) / hist.sum()
    fwiou = (freq[freq > 0] * iou[freq > 0]).sum()
    return iou, dice, recall, pre


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def convert_LeakyReLU_model(module):
    """
    类似 apex 里的 convert_syncbn_model 函数
    将 model 中的所有 ReLU 激活函数替换为 LeakyReLU
    :param module: 输入的模型
    :return: 转换后的模型
    """
    mod = module
    if isinstance(module, nn.ReLU):
        mod = nn.LeakyReLU(negative_slope=0.2, inplace=module.inplace)
    for name, child in module.named_children():
        mod.add_module(name, convert_LeakyReLU_model(child))

    return mod


def main():
    transforms_train = transforms.Compose([  # transforms.Resize(448),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transforms_val = transforms.Compose([  # transforms.Resize(448),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    voc_data = {'train': GTAdataset(root_dir=voc_root, train=True,
                                    trsf=transforms_train),
                'val': GTAdataset(root_dir=voc_root, train=False,
                                  trsf=transforms_val)}
    dataloaders = {'train': DataLoader(voc_data['train'], batch_size=1,
                                       shuffle=True, num_workers=8),
                   'val': DataLoader(voc_data['val'], batch_size=1,
                                     shuffle=False, num_workers=8)}
    dataset_sizes = {x: len(voc_data[x]) for x in ['train', 'val']}

    num_category = 3
    DDGAN = ddnet(3)

    num_epoch = 55
    criterion = nn.NLLLoss(ignore_index=255)

    optimizerG = optim.SGD(DDGAN.parameters(), lr=1e-3, momentum=0.99)  #/

    exp_lr_scheduler = lr_scheduler.StepLR(optimizerG, step_size=1765, gamma=0.9)  #
    myDDGAN = nn.DataParallel(DDGAN).cuda()
    since = time.time()
    # %% Train
    for t in range(num_epoch):  #

        myDDGAN.train()  # Set model to training mode
        tbar = tqdm(dataloaders['train'])
        running_lossg = 0
        running_lossG = 0

        # Iterate over data.
        for i, sample in enumerate(tbar):
            exp_lr_scheduler.step()
            inputs, labels, label_rgb = sample['image'], sample['label'], sample['label_rgb']
            inputs = inputs.cuda()
            labels = labels.cuda()
            label_rgb = label_rgb.cuda()
            optimizerG.zero_grad()
            with torch.set_grad_enabled(True):
                output = myDDGAN(inputs)
                predict_prob = F.log_softmax(output, dim=1)  # predict label
                loss_fp = criterion(predict_prob, labels.long())
                lossG = loss_fp
                lossG.backward()
                optimizerG.step()
            running_lossg += loss_fp.item() * inputs.size(0)
            running_lossG += lossG.item() * inputs.size(0)
        lossg = running_lossg / dataset_sizes['train']
        lossG_ad = running_lossG / dataset_sizes['train']
        print('Training Results({}): '.format(t))
        print('generator loss: {:4f}'.format(lossg))
        print('Generate ad loss:{:4f}'.format(lossG_ad))
        if (t + 1) % 1 == 0:
            torch.save(myDDGAN, '1030_test_rvot_pyt/1208_four/Conv_0910_12081_cross_chuncsc4_rvot_vgg16/original_method1_%d.pkl' % (t + 1))

# %%
if __name__ == '__main__':
    main()
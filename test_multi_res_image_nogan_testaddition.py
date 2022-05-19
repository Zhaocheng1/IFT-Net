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
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob
from skimage import segmentation as seg

datadir = '/home/guolibao/cardiac-jbhi/'
voc_root = os.path.join(datadir, 'cardiac-4ch')
os.environ['CUDA_VISIBLE_DEVICES']='1'


if torch.cuda.is_available():
    device=torch.device('cuda')
else:
    device=torch.device('cpu')
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

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
        image = transforms.Resize((256, 256), interpolation=Image.BILINEAR)(image)
        label = transforms.Resize((256, 256), interpolation=Image.BILINEAR)(label)
        sample = {'image': image, 'label': label}
        if self.trsf:
            sample = self.trsf(sample)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = transforms.ToTensor()(image)
        label = torch.from_numpy(np.array(label, dtype='int'))
        return {'image': image, 'label': label}


class Normalize(object):
    def __init__(self, mean=[0., 0., 0.], std=[1., 1., 1.]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = transforms.Normalize(self.mean, self.std)(image)
        return {'image': image, 'label': label}


# ?? bilinear kernel
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
    #miou = np.nanmean(iou)
    with np.errstate(divide='ignore', invalid='ignore'):
        dice = 2 * np.diag(hist) / (hist.sum(axis=0) + hist.sum(axis=1))
    #mdice = np.nanmean(dice)
    with np.errstate(divide='ignore', invalid='ignore'):
        recall = np.diag(hist) / hist.sum(axis=1)
    #mrecall = np.nanmean(recall)

    with np.errstate(divide='ignore', invalid='ignore'):
        spec = (np.diag(hist).sum() - np.diag(hist))/ ((np.diag(hist).sum() - np.diag(hist)) + (hist.sum(axis=0)-np.diag(hist)))

    with np.errstate(divide='ignore', invalid='ignore'):
        pre = np.diag(hist) / hist.sum(axis=0)
    #mpre = np.nanmean(pre)
    freq = hist.sum(axis=0) / hist.sum()
    fwiou = (freq[freq > 0] * iou[freq > 0]).sum()
    return  acc,iou, dice, recall, spec, pre

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
since = time.time()
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

    myfcn_gan = torch.load('/home/guolibao/train_addition/Comparison_experiment/12081/12081_four/Conv_0910_12081_cross_chuncsc4_rvot_vgg16/original_method1_15.pkl')

    # colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0]]
    # cm = np.array(colormap, dtype='uint8')
    # imgs = glob.glob('/home/guolibao/cardiac-jbhi/cardiac-4ch/val/*.png')
    # # imgs = glob.glob('/home/guolibao/cardiac-jbhi/cardiac-4ch/cardiac-rvot/val/*.png')
    # for i, img in enumerate(imgs):
    #     val_sample = voc_data['val'][i]
    #     val_image = val_sample['image'].cuda()
    #     val_label = val_sample['label']
    #     val_output = myfcn_gan(val_image.unsqueeze(0))
    #     val_pred = val_output.max(dim=1)[1].squeeze(0).data.cpu().numpy()
    #     val_label = val_label.long().data.numpy()
    #     val_image = val_image.squeeze().data.cpu().numpy().transpose((1, 2, 0))
    #     val_image = val_image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    #     val_image *= 255
    #     val_image = val_image.astype(np.uint8)
    #     val_pred_tian = cm[val_pred]
    #     val_pred_bin = seg.mark_boundaries(val_image, val_label, color=(1, 0, 0))
    #     val_pred_bin = seg.mark_boundaries(val_pred_bin, val_pred, color=(0, 1, 0))
    #     plt.imsave(osp.join('/home/guolibao/train_addition/Comparison_experiment/12081/Conv_0910_12081_cross_chuncsc4_rvot_vgg16/bin4', img.split('/')[-1]), val_pred_tian)
    #     plt.imsave(osp.join('/home/guolibao/train_addition/Comparison_experiment/12081/Conv_0910_12081_cross_chuncsc4_rvot_vgg16/pic4', img.split('/')[-1]), val_pred_bin)


    running_acc = 0

    # dice
    Dice_Lv = 0
    Dice_La = 0
    Dice_mean = 0
    # Recall
    Recall_Lv = 0
    Recall_La = 0
    Recall_mean = 0
    # Precision
    Precision_Lv = 0
    Precision_La = 0
    Precision_mean = 0
    #spec
    Spec_Lv = 0
    Spec_La = 0
    Spec_mean = 0
    # jacc
    Jacc_Lv = 0
    Jacc_La = 0
    Jacc_mean = 0

    for sample in tqdm(dataloaders['val']):
        # res_rec=[]
        inputs, labels = sample['image'], sample['label']
        inputs = inputs.cuda()
        labels = labels.cuda()
        # forward
        outputs = myfcn_gan(inputs)
        outputs = F.log_softmax(outputs, dim=1)
        preds = outputs.data.cpu().numpy()
        labels = labels.data.cpu().numpy()
        h, w = labels.shape[1:]
        ori_h, ori_w = preds.shape[2:]
        preds = np.argmax(ndimage.zoom(preds, (1., 1., 1. * h / ori_h, 1. * w / ori_w), order=1), axis=1)
        for pred, label in zip(preds, labels):
            # acc,macc,iou,fwiou,dice,recall,pre= evaluation_metrics(pred, label, num_category)
            acc,iou, dice, recall, spec, pre = evaluation_metrics(pred, label, num_category)
            # print(acc)
            running_acc += acc

            # compute lv la jacc,jacc shape [1,2,3],1 is background,2 is lv,3 is la
            jacc_la = iou[2:]  # pick up third figure
            jacc_la = np.mean(jacc_la)  # pick up number
            jaccll = iou[1:]  # pick up second and number figure
            jacc_mean = np.mean(jaccll)  # Mean second and third number
            jacc_lv = jaccll[:1]  # pikup first number of jaccll
            jacc_lv = np.mean(jacc_lv)  # pick up first number
            Jacc_Lv += jacc_lv  # add all
            Jacc_La += jacc_la
            Jacc_mean += jacc_mean
            # copute dice
            dice_la = dice[2:]
            dice_la = np.mean(dice_la)
            dice_ll = dice[1:]
            dice_mean = np.mean(dice_ll)
            dice_lv = dice_ll[:1]
            dice_lv = np.mean(dice_lv)
            Dice_La += dice_la
            Dice_Lv += dice_lv
            Dice_mean += dice_mean
            # compute recall
            # res_rec.append(recall)
            recall = np.nan_to_num(recall)
            recall_la = recall[2:]
            recall_la = np.mean(recall_la)
            recall_ll = recall[1:]
            recall_mean = np.mean(recall_ll)
            recall_lv = recall_ll[:1]
            recall_lv = np.mean(recall_lv)
            Recall_La += recall_la
            Recall_Lv += recall_lv
            Recall_mean += recall_mean
            #compute spec
            spec = np.nan_to_num(spec)
            spec_la = spec[2:]
            spec_la = np.mean(spec_la)
            spec_ll = spec[1:]
            spec_mean = np.mean(spec_ll)
            spec_lv = spec_ll[:1]
            spec_lv = np.mean(spec_lv)
            Spec_La += spec_la
            Spec_Lv += spec_lv
            Spec_mean += spec_mean
            # compute precision
            precision_la = pre[2:]
            precision_la = np.mean(precision_la)
            precision_ll = pre[1:]
            precision_mean = np.mean(precision_ll)
            precision_lv = precision_ll[:1]
            precision_lv = np.mean(precision_lv)
            Precision_La += precision_la
            Precision_Lv += precision_lv
            Precision_mean += precision_mean

    val_acc = running_acc / dataset_sizes['val']

    val_jacc_lv = Jacc_Lv / dataset_sizes['val']
    val_jacc_la = Jacc_La / dataset_sizes['val']
    val_jacc_mean = Jacc_mean / dataset_sizes['val']

    val_dice_lv = Dice_Lv / dataset_sizes['val']
    val_dice_la = Dice_La / dataset_sizes['val']
    val_dice_mean = Dice_mean / dataset_sizes['val']

    val_recall_lv = Recall_Lv / dataset_sizes['val']
    val_recall_la = Recall_La / dataset_sizes['val']
    val_recall_mean = Recall_mean / dataset_sizes['val']

    val_spec_lv = Spec_Lv / dataset_sizes['val']
    val_spec_la = Spec_La / dataset_sizes['val']
    val_spec_mean = Spec_mean / dataset_sizes['val']

    val_pre_lv = Precision_Lv / dataset_sizes['val']
    val_pre_la = Precision_La / dataset_sizes['val']
    val_pre_mean = Precision_mean / dataset_sizes['val']

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Validation Results: ')
    print('Pixel accuracy: {:4f}'.format(val_acc))
    # print('Mean accuracy: {:4f}'.format(val_macc))

    print('Jacc_lv acc:{:4f}'.format(val_jacc_lv))
    print('Jacc_la acc:{:4f}'.format(val_jacc_la))
    print('Jacc_mean acc:{:4f}'.format(val_jacc_mean))
    print('Dice_lv acc:{:4f}'.format(val_dice_lv))
    print('Dice_la acc:{:4f}'.format(val_dice_la))
    print('Dice_mean acc:{:4f}'.format(val_dice_mean))
    print('Recall_lv acc:{:4f}'.format(val_recall_lv))
    print('Recall_la acc:{:4f}'.format(val_recall_la))
    print('Recall_mean acc:{:4f}'.format(val_recall_mean))
    print('spec_lv acc:{:4f}'.format(val_spec_lv))
    print('spec_la acc:{:4f}'.format(val_spec_la))
    print('spec_mean acc:{:4f}'.format(val_spec_mean))
    print('Precision_lv acc:{:4f}'.format(val_pre_lv))
    print('Precision_la acc:{:4f}'.format(val_pre_la))
    print('Precision_mean acc:{:4f}'.format(val_pre_mean))


# %%
if __name__ == '__main__':
    main()
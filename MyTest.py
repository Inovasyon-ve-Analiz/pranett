import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.PraNet_Res2Net import PraNet
from utils.dataloader import test_dataset
import cv2

parser = argparse.ArgumentParser()
model_path = "models/PraNet-2.pth" #consists of three folders named MASKS, PNG and RESULTS
dataset_path = "dataset/TRAINING/KANAMA"
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default=model_path)

for _data_name in ['PNG']:#nothing but one single loop
    data_path = dataset_path  #'./KANAMA'#/{}/'.format(_data_name)
    save_path = '{}/RESULTS/'.format(data_path)    #dataset_path+'/results/'#.format(_data_name)
    opt = parser.parse_args()
    model = PraNet()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/PNG/'.format(data_path)
    gt_root = '{}/MASKS/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res5, res4, res3, res2 = model(image)
        res = res2
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path+name, res)#misc.imsave(save_path+name, res)

for i,f in enumerate(os.listdir(save_path)):
    img = cv2.imread(os.path.join(save_path,f),0)
    img[img>0]=255
    cv2.imwrite(os.path.join(save_path,f),img)
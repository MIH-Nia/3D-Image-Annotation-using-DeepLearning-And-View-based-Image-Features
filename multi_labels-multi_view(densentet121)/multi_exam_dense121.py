import os

import torch
from torch.autograd import Variable
from torchvision import transforms, models
from torchvision.datasets.folder import default_loader

import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



#from utils import make_classes

from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open('C:/Users/mh661/PycharmProjects/Image-Tagging-Model/data_processing/tag_multi_1.txt') as all_tags:
        tag_list = [item.rstrip() for item in all_tags.readlines()]

    number_of_tags = len(tag_list)
    # number_of_tags = 1161
    num_views=12

                         
    model_ft = models.densenet121(weights=models.densenet.DenseNet121_Weights.DEFAULT)
    model_ft.features[0] = nn.Conv2d(num_views, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, number_of_tags)
    model = nn.Sequential(model_ft, nn.Sigmoid()).to(device)

    model.load_state_dict(torch.load("C:/Users/mh661/PycharmProjects/Image-Tagging-Model/models/densenet121-2.ckpt",map_location = torch.device('cpu')))
    was_training = model.training
    model.eval()

    with torch.no_grad():
        fig = plt.figure()
        train_dataset_m = MultiviewImgDataset(root_dir='C:/Users/mh661/Desktop/final_mv_mtag/image_12/bench/test', scale_aug=False, rot_aug=False,test_mode=True, num_models=0,  num_views=12,shuffle=False )
        imgs = DataLoader(train_dataset_m, batch_size=12, shuffle=False, num_workers=2)
        threshold = 0.6
        for data in tqdm(imgs):
          inputt = data['image'].to(device)
        outputs = model(inputt)
        out_np=np.array(outputs)
        t = Variable(torch.Tensor([threshold])).to(device)
        preds = (outputs > t).float() * 1
        preds = preds.type(torch.IntTensor).cpu().numpy()
        out_np = np.array(preds)
        tagss=[]
        for i in range(len(train_dataset_m)):
            print('_____________________________________________________________')
            for j in range(len(tag_list)):
                if out_np[i][j]==1:
                    tagss.append(tag_list[j])
            print('shape_%i :' %(i+1)); print(tagss)
            tagss = []

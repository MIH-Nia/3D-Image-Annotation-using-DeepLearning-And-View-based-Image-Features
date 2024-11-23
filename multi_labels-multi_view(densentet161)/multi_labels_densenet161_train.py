
from __future__ import print_function, division

import copy
import os
import time

from numpy import array
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, utils, models
import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.autograd import Variable



from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset


def show_batch(sample_batched):
    """Show image for a batch of samples."""
    images_batch, labels_batch = sample_batched['image'], sample_batched['labels']
    batch_size = len(images_batch)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.title('Batch from dataloader')


def show_image(img):
    """Show image for a batch of samples."""
    plt.imshow(img.numpy().transpose((1, 2, 0)))


def train_model(model, train, val, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_loss = 100.0
    train_error_list = list()
    val_error_list = list()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train
                model.train()  # Set model to training mode
            else:
                dataloader = val
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            num_views = 12
            rand_idx = np.random.permutation(int(len(dataloader.dataset.filepaths) / num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(dataloader.dataset.filepaths[
                                       rand_idx[i] * num_views:(rand_idx[i] + 1) * num_views])
            dataloader.dataset.filepaths = filepaths_new

            # Iterate over data.
            for data in tqdm(dataloader):
                inputs, labels = data['image'].to(device), data['labels'].type(
                    torch.FloatTensor).to(device)
                print("_____________________________________________________________________")
                print("input_shape:" ,inputs.shape)
                print("labels_shape:", labels.shape)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    print("outputs_shape:" , outputs.shape)
                    loss = criterion(outputs, labels)
                    print("loss :",loss*100)
                    print("_____________________________________________________________________")
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                # statistics
                running_loss += loss.item() * 100

            epoch_loss = running_loss / len(dataloader)

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            if phase == 'train':
                train_error_list.append(epoch_loss)
            else:
                val_error_list.append(epoch_loss)
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), 'C:/Users/mh661/Desktop/multi_labels-multi_view_10_classes/models/densenet161_cpk/densenet161-{}.ckpt'.format(epoch + 1))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_loss))
    print('Train_error_list:', train_error_list)
    print('Val_error_list:', val_error_list)

    # load best model weights
    # model.load_state_dict(best_model_wts)
    # return model

if __name__ == "__main__":
    with open('C:/Users/mh661/Desktop/multi_labels-multi_view_10_classes/labels_10.txt') as all_tags:
         tag_list = [item.rstrip() for item in all_tags.readlines()]
         print("Classes: ", tag_list)
    number_of_tags = len(tag_list)
    num_view=12
    train_dataset_m = MultiviewImgDataset(root_dir='C:/Users/mh661/Desktop/modelnet10_reduce/*/train', scale_aug=False, rot_aug=False, num_models=0,  num_views=12 )
    train_dataset_m.filepaths=np.array(train_dataset_m.filepaths)
    train_dataset_m.classnames = np.array(train_dataset_m.classnames)
    train_loader_m = DataLoader(train_dataset_m, batch_size=32, shuffle=True, num_workers=4)


    val_dataset_m = MultiviewImgDataset(root_dir='C:/Users/mh661/Desktop/modelnet10_reduce/*/test', scale_aug=False, rot_aug=False, num_views=12 )
    val_dataset_m.filepaths=np.array(val_dataset_m.filepaths)
    val_dataset_m.classnames = np.array(val_dataset_m.classnames)

    val_loader_m =DataLoader(val_dataset_m, batch_size=32, shuffle=False, num_workers=4)
    print('num_train_files: ' + str(len(train_dataset_m.filepaths)))
    print('num_val_files: ' + str(len(val_dataset_m.filepaths)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = models.densenet161(weights=models.densenet.DenseNet161_Weights.DEFAULT)
    model_ft.features[0] = nn.Conv2d(num_view, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, number_of_tags)
    model = nn.Sequential(model_ft, nn.Sigmoid()).to(device)

    #model.load_state_dict(torch.load("C:/Users/mh661/Desktop/multi_labels-multi_view_10_classes/models/densenet161_cpk/densenet161-4.ckpt"))  # 0.7
    for param in model[0].parameters():
        param.requires_grad = False
    for param in model[0].features.denseblock3.parameters():
        param.requires_grad = True
    for param in model[0].features.transition3.parameters():
        param.requires_grad = True
    for param in model[0].features.denseblock4.parameters():
        param.requires_grad = True
    for param in model[0].classifier.parameters():
        param.requires_grad = True

    criterion = nn.BCELoss()
    optimizer_ft = optim.Adam(model.parameters(), lr=0.0003 ,weight_decay=0 ,betas=(0.9, 0.999))
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

    train_model(model, train_loader_m, val_loader_m, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=2)
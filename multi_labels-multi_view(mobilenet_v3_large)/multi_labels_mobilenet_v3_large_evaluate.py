import time

import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from tools.ImgDataset import MultiviewImgDataset

def F_score(ground_truth, predicted):
    true_positives = set(ground_truth).intersection(set(predicted))
    false_positives = set(predicted) - set(true_positives)
    false_negatives = set(ground_truth) - set(true_positives)
    return len(true_positives), len(false_positives), len(false_negatives)

if __name__ == "__main__":
    # start = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open('C:/Users/mh661/PycharmProjects/multi label_multi veiw/data_processing/tag_multi_1.txt') as all_tags:
         tag_list = [item.rstrip() for item in all_tags.readlines()]
         print("Classes: ", tag_list)
    number_of_tags = len(tag_list)
    num_view=12
    number_of_tags = len(tag_list)
    model_ft = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    model_ft.features[0] = nn.Conv2d(num_view, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model_ft.features[16]=nn.Sequential(nn.Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False),nn.BatchNorm2d(960, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),nn.SiLU(inplace=True))
    num_ftrs = model_ft.classifier[3].in_features
    model_ft.classifier[3] = nn.Linear(num_ftrs, number_of_tags)
    model = nn.Sequential(model_ft, nn.Sigmoid()).to(device)


    model.load_state_dict(torch.load("C:/Users/mh661/PycharmProjects/multi label_multi veiw/models/mobilenet_v3_large_cpk/mobilenet_v3_large-35.ckpt"))
    model.eval()

    test_dataset = MultiviewImgDataset(root_dir='C:/Users/mh661/Desktop/test_eval/*/test', scale_aug=False,
                                          rot_aug=False, num_models=0, num_views=12)
    test_dataset.filepaths = np.array(test_dataset.filepaths)
    test_dataset.classnames = np.array(test_dataset.classnames)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    for param in model[0].parameters():
        param.requires_grad = False
    with torch.no_grad():
        # all_true_positives = 0
        # #بالایی می شود درست بوده و به عنوان درست تشخیص داده است.
        # all_false_positives = 0
        # #بالایی می شود غلط بوده و به عنوان درست تشخیص داده است.
        # all_false_negatives = 0
        # #بالایی می شود غلط بوده و به عنوان غلط تشخیص داده است.
        list_threshold = np.arange(0, 1, 0.05)
        list_F_score = list()

        for threshold in list_threshold:
            print("")
            all_true_positives = 0
            # بالایی می شود درست بوده و به عنوان درست تشخیص داده است.
            all_false_positives = 0
            # بالایی می شود غلط بوده و به عنوان درست تشخیص داده است.
            all_false_negatives = 0
            # بالایی می شود غلط بوده و به عنوان غلط تشخیص داده است.
            for sample_batched in tqdm(test_loader):
                inputs, labels = sample_batched['image'].to(device), sample_batched['labels']

                outputs = model(inputs)
                t = Variable(torch.Tensor([threshold])).to(device)
                preds = (outputs > t).float() * 1
                preds = preds.type(torch.IntTensor).cpu().numpy()

                for j in range(inputs.size()[0]):
                    pred_list = list()
                    ground_truth_list = list()
                    for num_item, item in enumerate(preds[j]):
                        if item != 0:
                            pred_list.append(tag_list[item * num_item])
                    for num_item, item in enumerate(labels[j]):
                        if item != 0:
                            ground_truth_list.append(tag_list[item * num_item])

                    true_positives, false_positives, false_negatives = F_score(ground_truth_list, pred_list)

                    all_true_positives += true_positives
                    all_false_positives += false_positives
                    all_false_negatives += false_negatives

            precision = all_true_positives/(all_true_positives+all_false_positives+1e-10)
            recall = all_true_positives/(all_true_positives+all_false_negatives+1e-10)
            F1 = 2*precision*recall/(precision+recall+1e-10)
            list_F_score.append(F1)
            print('\n')
            print("threshold :",threshold)
            print("precision :",precision)
            print("recall :",recall)
            print("F1 :",F1)
            print('____________________________________________________________________')

        plt.plot(list_threshold, list_F_score,color='r')
        plt.plot(list_threshold, list_F_score, 'g^',color='r')
        plt.show()

    # visualize_model(device, model, tag_list, test_loader, num_images=4)

    # test_image(device, model, tag_list, '/home/tthieuhcm/Downloads/image1.png')
    # print(time.time()-start)

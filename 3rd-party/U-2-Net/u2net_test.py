import os
import os.path
import argparse

import sys
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , image_processing

import numpy as np
from PIL import Image  # , ImageOps
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET  # full size version 173.6 MB
from model import U2NETP  # small version u2net 4.7 MB


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('L')  # .convert('RGB')) -- if you need 3-channel output
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    path = os.path.join(d_dir, imidx + '.png')
    imo.save(path)


def main():
    # --------- 1. get image path and name ---------
    model_name = 'u2net'  # u2netp

    image_dir_0 = os.path.join(os.getcwd(), 'test_data', 'test_images')
    prediction_dir_0 = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    model_dir_0 = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_models', model_name + '.pth')

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m", "--model_path", default=model_dir_0, help="pretrained model as .pth file"
    )
    parser.add_argument(
        "-i", "--input_path", default=image_dir_0, help="folder with input images"
    )
    parser.add_argument(
        "-o", "--output_path", default=prediction_dir_0, help="folder with output images"
    )

    args = parser.parse_args()

    image_dir = args.input_path
    prediction_dir = args.output_path
    pretrained_model_path = args.model_path

    if not os.path.exists(pretrained_model_path):
        print(f"Could not find pretrained U^2 Net model at {pretrained_model_path}. "
              f"Please run CMake to download the default pretrained model or specify path to the pretrained model"
              f" as the -m argument to the u2net_test script.")
        return -1

    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)

    # --------- 2. dataloader ---------
    # 1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if model_name == 'u2net':
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)
    elif model_name == 'u2netp':
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3, 1)
    else:
        raise ValueError(f"Unsuppored model name: {model_name}")

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(pretrained_model_path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(pretrained_model_path, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test], pred, prediction_dir)

        del d1, d2, d3, d4, d5, d6, d7
    return 0


if __name__ == "__main__":
    sys.exit(main())

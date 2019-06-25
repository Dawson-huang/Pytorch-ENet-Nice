import torch
import torch.nn as nn
from utils import *
from models.ENet import ENet
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

def test(FLAGS):
    # Check if the pretrained model is available
    if not FLAGS.m.endswith('.pth'):
        raise RuntimeError('Unknown file passed. Must end with .pth')
    if FLAGS.image_path is None or not os.path.exists(FLAGS.image_path):
        raise RuntimeError('An image file path must be passed')
    
    h = FLAGS.resize_height
    w = FLAGS.resize_width
    nc = FLAGS.num_classes
    test_mode = FLAGS.test_mode


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")##dawson add gpu mode
    checkpoint = torch.load(FLAGS.m,  map_location=lambda storage, loc: storage.cuda(0))#"cpu")


    # Assuming the dataset is camvid
    enet = ENet(nc)
    enet.to(device) ##dawson add gpu mode
    enet.load_state_dict(checkpoint['state_dict'])

    tmg_ = plt.imread(FLAGS.image_path)
    tmg_ = cv2.resize(tmg_, (h, w), cv2.INTER_NEAREST)
    tmg = torch.tensor(tmg_).unsqueeze(0).float()
    tmg = tmg.transpose(2, 3).transpose(1, 2)

    with torch.no_grad():
        out1 = enet(tmg.cuda().float()).squeeze(0)##dawson add gpu mode
    

    b_ = out1.data.max(0)[1].cpu().numpy()


    if test_mode.lower() == 'cityscapes':
        decoded_segmap = decode_segmap(b_, True)
    else:
        decoded_segmap = decode_segmap(b_, False)

    cv2.imshow("test", decoded_segmap)
    cv2.waitKey()

    images = {
        0 : ['Input Image', tmg_],
        1 : ['Predicted Segmentation', b_],
    }

    show_images(images)

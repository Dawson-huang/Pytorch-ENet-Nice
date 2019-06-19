import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os
from PIL import Image
import torch

def create_class_mask(img, color_map, is_normalized_img=True, is_normalized_map=False, show_masks=False):
    """
    Function to create C matrices from the segmented image, where each of the C matrices is for one class
    with all ones at the pixel positions where that class is present

    img = The segmented image

    color_map = A list with tuples that contains all the RGB values for each color that represents
                some class in that image

    is_normalized_img = Boolean - Whether the image is normalized or not
                        If normalized, then the image is multiplied with 255

    is_normalized_map = Boolean - Represents whether the color map is normalized or not, if so
                        then the color map values are multiplied with 255

    show_masks = Wherether to show the created masks or not
    """

    if is_normalized_img and (not is_normalized_map):
        img *= 255

    if is_normalized_map and (not is_normalized_img):
        img = img / 255
    
    mask = []
    hw_tuple = img.shape[:-1]
    for color in color_map:
        color_img = []
        for idx in range(3):
            color_img.append(np.ones(hw_tuple) * color[idx])

        color_img = np.array(color_img, dtype=np.uint8).transpose(1, 2, 0)

        mask.append(np.uint8((color_img == img).sum(axis = -1) == 3))

    return np.array(mask)


def loader(training_path, segmented_path, batch_size, h=512, w=512):
    """
    The Loader to generate inputs and labels from the Image and Segmented Directory

    Arguments:

    training_path - str - Path to the directory that contains the training images

    segmented_path - str - Path to the directory that contains the segmented images

    batch_size - int - the batch size

    yields inputs and labels of the batch size
    """

    filenames_t = os.listdir(training_path)
    total_files_t = len(filenames_t)
    
    filenames_s = os.listdir(segmented_path)
    total_files_s = len(filenames_s)
    
    assert(total_files_t == total_files_s)
    
    if str(batch_size).lower() == 'all':
        batch_size = total_files_s
    
    idx = 0
    while(1):
        batch_idxs = np.random.randint(0, total_files_s, batch_size)
            
        
        inputs = []
        labels = []
        
        for jj in batch_idxs:
            img = plt.imread(training_path + filenames_t[jj])
            #img = cv2.resize(img, (h, w), cv2.INTER_NEAREST)
            inputs.append(img)
            
            img = Image.open(segmented_path + filenames_s[jj])
            img = np.array(img)
            #img = cv2.resize(img, (h, w), cv2.INTER_NEAREST)
            labels.append(img)
         
        inputs = np.stack(inputs, axis=2)
        inputs = torch.tensor(inputs).transpose(0, 2).transpose(1, 3)
        
        labels = torch.tensor(labels)
        
        yield inputs, labels


def loader_cityscapes(txt_path, cityscapes_path, batch_size):
    """
    The Loader to generate inputs and labels from the txt file

    Arguments:

    txt_path - str - Path to the txt file that contains the training images and segmented images path

    cityscapes_path - str - Cityscapes Path to the directory of Cityscapes image

    batch_size - int - the batch size

    yields inputs and labels of the batch size
    """

    lines = open(txt_path, 'r').readlines()
    total_files = len(lines)

    images = []
    gts = []
    for line in lines:
        line = line.strip().split(" ")
        images.append(line[0])
        gts.append(line[1])

    if str(batch_size).lower() == 'all':
        batch_size = total_files
        while (1):
            batch_idxs = np.random.randint(0, total_files, batch_size)

            labels = []
            for jj in batch_idxs:
                img = Image.open(cityscapes_path + gts[jj])
                img = np.array(img)
                #img5 = scale_downsample(img, 0.5, 0.5)

                labels.append(img)

            labels = torch.tensor(labels)

            yield labels

    idx = 0
    while (1):
        batch_idxs = np.random.randint(0, total_files, batch_size)

        inputs = []
        labels = []
        for jj in batch_idxs:

            img = plt.imread(cityscapes_path + images[jj])
            #img5 = scale_downsample(img, 0.5, 0.5)
            inputs.append(img)

            img = Image.open(cityscapes_path + gts[jj])
            img = np.array(img)
            #img5 = scale_downsample(img, 0.5, 0.5)
            labels.append(img)

        inputs = np.stack(inputs, axis=2)
        inputs = torch.tensor(inputs).transpose(0, 2).transpose(1, 3)

        labels = torch.tensor(labels)

        yield inputs, labels



def decode_segmap(image, cityscapes):
    Sky = [128, 128, 128]
    Building = [128, 0, 0]
    Column_Pole = [192, 192, 128]
    Road_marking = [255, 69, 0]
    Road = [128, 64, 128]
    Pavement = [60, 40, 222]
    Tree = [128, 128, 0]
    SignSymbol = [192, 128, 128]
    Fence = [64, 64, 128]
    Car = [64, 0, 128]
    Pedestrain = [64, 64, 0]
    Bicyclist = [0, 128, 192]


    road = [128,64,128]
    Lsidewalk = [244,35,232]
    building = [70,70,70]
    wall = [102,102,156]
    fence = [190,153,153]
    pole = [153,153,153]
    traffic_light = [250,170,30]
    traffic_sign = [220,220,0]
    vegetation = [107,142,35]
    terrain = [152,251,152]
    sky = [70,130,180]
    person = [220,20,60]
    Lrider = [255,0,0]
    car = [0,0,142]
    truck = [0,0,70]
    bus = [0,60,100]
    train = [0,80,100]
    motorcycle = [0,0,230]
    bicycle = [119,11,32]


    if cityscapes:
        label_colors = np.array([road, Lsidewalk, building, wall, fence, pole, traffic_light, traffic_sign,
                                 vegetation, terrain, sky, person, Lrider, car, truck, bus, train, motorcycle,
                                 bicycle]).astype(np.uint8)
    else:
        label_colors = np.array([Sky, Building, Column_Pole, Road_marking, Road,
                              Pavement, Tree, SignSymbol, Fence, Car,
                              Pedestrain, Bicyclist]).astype(np.uint8)

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for label in range(len(label_colors)):
            b[image == label] = label_colors[label, 0]
            g[image == label] = label_colors[label, 1]
            r[image == label] = label_colors[label, 2]

    rgb = np.zeros((image.shape[0], image.shape[1], 3)).astype(np.uint8)
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b

    return rgb

def show_images(images, in_row=True):
    '''
    Helper function to show 3 images
    '''
    total_images = len(images)

    rc_tuple = (1, total_images)
    if not in_row:
        rc_tuple = (total_images, 1)

    #figure = plt.figure(figsize=(20, 10))
    for ii in range(len(images)):
        plt.subplot(*rc_tuple, ii+1)
        plt.title(images[ii][0])
        plt.axis('off')
        plt.imshow(images[ii][1])
    # plt.savefig("./Enet.png")
    plt.show()

def get_class_weights(loader, num_classes, c=1.02, isCityscapes=False):
    '''
    This class return the class weights for each class
    
    Arguments:
    - loader : The generator object which return all the labels at one iteration
               Do Note: That this class expects all the labels to be returned in
               one iteration

    - num_classes : The number of classes

    Return:
    - class_weights : An array equal in length to the number of classes
                      containing the class weights for each class
    '''
    if isCityscapes:
        labels = next(loader)
    else:
        _, labels = next(loader)
    all_labels = labels.flatten()
    all_len = len(all_labels)
    each_class = np.bincount(all_labels, minlength=num_classes)
    if isCityscapes:
        each_class = each_class[0:19]
        num = 0
        for i in each_class:
            num += i
        all_len = num
    prospensity_score = each_class / all_len
    class_weights = 1 / (np.log(c + prospensity_score))
    print("class_weights: ")
    print(class_weights)
    return class_weights


def scale_downsample(img, kx, ky):
    rows = int(np.round(np.abs(img.shape[0] * kx)))
    cols = int(np.round(np.abs(img.shape[1] * ky)))

    if len(img.shape) == 3 and img.shape[2] >= 3:
        dist = np.zeros((rows, cols, img.shape[2]), img.dtype)
    else:
        dist = np.zeros((rows, cols), img.dtype)

    for y in range(rows):
        for x in range(cols):
            new_y = int((y + 1) / ky + 0.5) - 1
            new_x = int((x + 1) / kx + 0.5) - 1

            dist[y, x] = img[new_y, new_x]

    return dist
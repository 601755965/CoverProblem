from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

# from PIL import Image

import torch
from torch.utils.data import DataLoader
# from torchvision import datasets
from torch.autograd import Variable

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from matplotlib.ticker import NullLocator



if __name__ == "__main__":

# def test(img_path, anno_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_90_0.904.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--img_path", type=str, default="", help="path of image")
    parser.add_argument("--anno_path", type=str, default="", help="path of label")
    opt = parser.parse_args()
    # opt.img_path = img_path
    # opt.anno_path = anno_path
    print(opt)


    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device("cpu")
    # os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    model = torch.load(opt.weights_path, map_location='cpu')

    # if opt.weights_path.endswith(".weights"):
    #     # Load darknet weights
    #     model.load_darknet_weights(opt.weights_path)
    # else:
    #     # Load checkpoint weights
    #     model.load_state_dict(torch.load(opt.weights_path, map_location='cpu'))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.img_path, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    # Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    Tensor = torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    # cmap = plt.get_cmap("tab20b")
    # colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    # print("\nSaving images:")
    # Iterate through images and save plot of detections

    coreStr = ''
    corelessStr = ''

    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        # print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        # plt.figure()
        # fig, ax = plt.subplots(1)
        # ax.imshow(img)
        # print(detections)
        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            # bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                # color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                # bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                # ax.add_patch(bbox)
                # Add label

                # name =  '%s %.2f \n (%.1f,%.1f) (%.1f,%.1f)' % (classes[int(cls_pred)], conf,x1,y1,x2,y2)
                # name = '%s %.2f ' % (classes[int(cls_pred)], conf)

                label = str(classes[int(cls_pred)])

                imgName = path.split("/")[-1].split(".")[0]

                if float(conf) >= 0.01:
                    if label == 'core':
                        coreStr += '%s %.3f %.1f %.1f %.1f %.1f\n' % (imgName, conf, x1, y1, x2, y2)
                    else:
                        corelessStr += '%s %.3f %.1f %.1f %.1f %.1f\n' % (imgName, conf, x1, y1, x2, y2)
                # print(label, imgName, conf, x1, y1, x2, y2)




                # plt.text(
                #     x1,
                #     y1,
                #     s=name,
                #     color="white",
                #     verticalalignment="baseline",
                #     bbox={"color": color, "pad": 0},
                # )

        # Save generated image with detections
        # plt.axis("off")
        # plt.gca().xaxis.set_major_locator(NullLocator())
        # plt.gca().yaxis.set_major_locator(NullLocator())
        # filename = path.split("/")[-1].split(".")[0]
        # plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        # plt.close()

    # print('--------')
    # print(corelessStr)
    # print('--------')
    # print(coreStr)

    f = open('../predicted_file/det_test_带电芯充电宝.txt', 'w')
    f.write(coreStr)
    f.close()

    f = open('../predicted_file/det_test_不带电芯充电宝.txt', 'w')
    f.write(corelessStr)
    f.close()

    print('\nlabels save as :\n')
    path1 = os.path.realpath('../predicted_file/det_test_带电芯充电宝.txt')
    print('\t', path1)
    path2 = os.path.realpath('../predicted_file/det_test_不带电芯充电宝.txt')
    print('\t', path2, '\n')

# test('/Users/xuefeng/Desktop/测试脚本/Image_test','/Users/xuefeng/Desktop/测试脚本/Anno_test')

# python test.py --img_path=/Users/xuefeng/Desktop/测试脚本/Image_test --anno_path /Users/xuefeng/Desktop/测试脚本/Anno_test
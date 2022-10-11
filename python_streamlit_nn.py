
#import streamlit as st
import torch
import detectron2
from detectron2.utils.logger import setup_logger

import numpy as np
import time
import os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import ColorMode
import datetime
from datetime import timedelta
import shutil
from distutils import file_util, dir_util
from distutils.dir_util import copy_tree
import glob
from contextlib import redirect_stdout
import tempfile
import statistics
from scipy.spatial import distance as dist
import imutils
from imutils import perspective
from imutils import contours
import csv
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
from PIL import Image
import seaborn as sns

setup_logger()

st.title('Deploying a Pytorch model on Streamlit | Detectron2')
st.write('What is Detectron2?')
st.write('Detectron2 is Facebook AI Researchs next generation software system that implements state-of-the-art object detection algorithms. It is a ground-up rewrite of the previous version, Detectron, and it originates from maskrcnn-benchmark.')
st.image('assets/img.png')

# import some common detectron2 utilities

st.write('\n')

st.title('Testing the Zoo Model')
st.write('Test image')
im = cv2.imread("assets/test_image1.jpeg")
# showing image
st.image('assets/test_image1.jpeg')

# import Trainer & init cfg
cfg = get_cfg()

# custom dataloader

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=[
            T.RandomBrightness(0.75, 1.25),
            T.RandomContrast(0.75, 1.25),
            T.RandomSaturation(0.75, 1.25),
            T.RandomLighting(0.75, 1.25),
            T.RandomRotation(angle=[-90,90], expand = False),                                   
            T.RandomApply(T.RandomFlip(horizontal=True, vertical=False), prob = 0.5),
            T.RandomApply(T.RandomFlip(horizontal=False, vertical=True), prob = 0.5),
            T.RandomApply(T.RandomFlip(horizontal=True, vertical=True), prob = 0.5),
            ])
        return build_detection_train_loader(cfg, mapper=mapper)

# setting of a training
cfg.MODEL.DEVICE = "cuda"                                                                                 # if working without GPU, write cpu instead of cuda

#cond for diff models                                                                              

cfg.DATASETS.TRAIN = ("manittol_s_train",)                                                                # name of the training dataset from registration
cfg.DATASETS.TEST = ("manittol_s_test",)                                                                  # name of the test dataset from registration
cfg.DATALOADER.NUM_WORKERS = 2                                                                            # tutor setting

#cond for diff models                                                                              
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2                                                                              # tutor setting
cfg.SOLVER.BASE_LR = 0.00025                                                                              # taken as "optimised" setting
cfg.SOLVER.MAX_ITER = 3000                                                                                # max number of iterations
cfg.SOLVER.STEPS = []                                                                                     # solver steps decay
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64                                                             # default=512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3                                                                       # number of annotated classes - 1 (crystals)

# def out
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)

# def predicted model
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, r"/content/output/model_final.pth")  # this calls for trained network  
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5                                           # treshold above which it takes it as a crystal
cfg.MODEL.DEVICE = "cuda"
predictor = DefaultPredictor(cfg)

# Catalogs
MetadataCatalog.get("manittol_s_test").set(things_classes=["Particle","Bubble","Droplet"])
MetadataCatalog.get("manittol_s_test").set(things_colors=[(0,0,225),(0,255,0),(255,0,0)])
manittol_s_test_metadata = MetadataCatalog.get("manittol_s_test")

# # inference
# im = cv2.imread(r"/content/drive/MyDrive/Colab Notebooks/DATA/Manittol/Manittol_crystals/Compare/Network/1.png")
# outputs = predictor(im)
# v = Visualizer(im[:, :, ::-1], 
#                 metadata=manittol_s_test_metadata,
#                 scale=1, 
#                 instance_mode=ColorMode.SEGMENTATION)                                  
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2_imshow(out.get_image()[:, :, ::-1])

# inference

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def GetInference():
  outputs = predictor(im)
  v = Visualizer(im[:, :, ::-1], 
                  metadata=manittol_s_test_metadata,
                  scale=1, 
                  instance_mode=ColorMode.SEGMENTATION)                                  
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  cv2_imshow(out.get_image()[:, :, ::-1])

def GetCounts():
  outputs = predictor(im)
  classes = outputs["instances"].pred_classes.to("cpu").numpy()
  TotalCount = sum(classes==1)+sum(classes==2)+sum(classes==3)
  ParticleCount = sum(classes==1)
  BubbleCount = sum(classes==2)
  DropletCount = sum(classes==3)
  # print("Total Count:  " + repr(TotalCount) + "\n" + "  Particle Count:  " + repr(ParticleCount) + "\n" + "  Bubble Count:  " + repr(BubbleCount) + "\n" + "  Droplet Count:  " + repr(DropletCount))
  
  ### Visualize Individual Mask ########################################

  PList.append(ParticleCount)
  DList.append(DropletCount)
  BList.append(BubbleCount)

def GetMask():
  outputs = predictor(im)
  mask_array = outputs['instances'].pred_masks.to("cpu").numpy()
  num_instances = mask_array.shape[0]
  mask_array = np.moveaxis(mask_array, 0, -1)
  mask_array_instance = []
  output = np.zeros_like(im) #black

  #print('output',output)


  fig = plt.figure(figsize=(15, 20))
  for i in range(num_instances):
      mask_array_instance.append(mask_array[:, :, i:(i+1)])
      output = np.where(mask_array_instance[i] == True, 255, output)

  imm = Image.fromarray(output)
  imm.save('predicted_masks.jpg')

  #cv2.imwrite('C:/Users/akeem.olaleye/detection/Manitol_PSD_ipynb/Masks' + '/' + 'Masks.jpg', output) #mask
  cv2.imwrite('Masks.jpg', output) #mask
  opening = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)


def GetContours():
  # find contours in the edge map
  cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)

  # sort the contours from left-to-right and initialize the
  # 'pixels per metric' calibration variable
  (cnts, _) = contours.sort_contours(cnts)
  pixelsPerMetric = 0.85

  # loop over the contours individually
  for c in cnts:

      # if the contour is not sufficiently large, ignore it
      if cv2.contourArea(c) < 100:
          continue

      # compute the rotated bounding box of the contour
      area = cv2.contourArea(c)
      perimeter = cv2.arcLength(c, True)

      orig = opening.copy()
      box = cv2.minAreaRect(c)
      box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
      box = np.array(box, dtype="int")
      # order the points in the contour such that they appear
      # in top-left, top-right, bottom-right, and bottom-left
      # order, then draw the outline of the rotated bounding
      # box
      box = perspective.order_points(box)
      cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

      # loop over the original points and draw them
      for (x, y) in box:
          cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
          # unpack the ordered bounding box, then compute the midpoint
          # between the top-left and top-right coordinates, followed by
          # the midpoint between bottom-left and bottom-right coordinates

      (tl, tr, br, bl) = box
      (tltrX, tltrY) = midpoint(tl, tr)
      (blbrX, blbrY) = midpoint(bl, br)

      # compute the midpoint between the top-left and top-right points,
      # followed by the midpoint between the top-righ and bottom-right
      (tlblX, tlblY) = midpoint(tl, bl)
      (trbrX, trbrY) = midpoint(tr, br)

          # compute the Euclidean distance between the midpoints
      dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
      dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

      # if the pixels per metric has not been initialized, then
      # compute it as the ratio of pixels to supplied metric
      # (in this case, inches)
      if pixelsPerMetric is None:
          pixelsPerMetric = dB / width

      # compute the size of the object
      dimA = dA / pixelsPerMetric
      dimB = dB / pixelsPerMetric
      dimArea = area/pixelsPerMetric
      dimPerimeter = perimeter/pixelsPerMetric
      diaFeret = max(dimA, dimB)

      if (dimA and dimB) !=0:
          Aspect_Ratio = max(dimB,dimA)/min(dimA,dimB)
      else:
          Aspect_Ratio = 0

      Length = min(dimA, dimB)
      Width = max(dimA, dimB)
      CircularED = np.sqrt(4*area/np.pi)
      Chords = cv2.arcLength(c,True)
      Roundness = 1/(Aspect_Ratio) if Aspect_Ratio != 0 else 0   # inverse of aspect ratio
      Sphericity = (2*np.sqrt(np.pi*dimArea))/dimPerimeter #Circularity squared
      Circularity = 4*np.pi*(dimArea/(dimPerimeter)**2)
      Feret_diam = diaFeret
      # csvUser = open('ShapeDescriptor.csv', 'a', newline='', encoding='utf8')
      # usersWriter = csv.writer(csvUser)
      # usersWriter.writerow([Feret_diam,Aspect_Ratio,Roundness,Circularity,Sphericity,Length,Width,CircularED,Chords],)
      # csvUser.close()

      lengthList.append(Length)
      widthList.append(Width)
      circularEDList.append(CircularED)
      aspectRatioList.append(Aspect_Ratio)
      circularityList.append(Circularity)
      chordsList.append(Chords) 


  # df = pd.read_csv('ShapeDescriptor.csv', header=None)
  # df.columns = ['Feret Diameter', 'Aspect Ratio', 'Roundness', 'Circularity', 'Sphericity', 'Length', 'Width', 'CircularED', 'Chords']
  # df.to_csv('Results.csv', index=True)

  ################ Histogram/Kernel Density of Crystal Shape/Size Properties ##################

  # sns.displot(df['Feret Diameter'])
  # sns.displot(df['Aspect Ratio'])
  # sns.displot(df['Roundness'])
  # sns.displot(df['Circularity'])
  # sns.displot(df['Sphericity'])
  # sns.displot(df['CircularED'])
  # sns.displot(df['Chords'])
  # sns.displot(df['Feret Diameter'], kind='kde')
  # sns.displot(df['Aspect Ratio'], kind='kde')
  # sns.displot(df['Roundness'], kind='kde')
  # sns.displot(df['Circularity'], kind='kde')
  # sns.displot(df['Sphericity'], kind='kde')
  # sns.displot(df['CircularED'], kind='kde')
  # sns.displot(df['Chords'], kind='kde')

  ## Bins

lengthList = list()
widthList = list()
circularEDList = list()
aspectRatioList = list()
circularityList = list()
chordsList = list()
PList = list()
DList = list()
BList = list()
tPL = 0
tBL = 0
tDL = 0


test_imgs = ['1','2','3','4','5','6','7','8','9','10']
for test_img in test_imgs:
  im = cv2.imread(r"/content/output/test/" + test_img + ".png")

  GetInference()
  GetMask()
  GetCounts()
  GetContours()

lengthBins = np.histogram(np.asarray(lengthList))
widthBins = np.histogram(np.asarray(widthList))
circularEDBins = np.histogram(np.asarray(circularEDList))
aspectRatioBins = np.histogram(np.asarray(aspectRatioList))
circularityBins = np.histogram(np.asarray(circularityList))
chordsBins = np.histogram(np.asarray(chordsList))

for PL in range(0, len(PList)):
    tPL = tPL + PList[PL]
for DL in range(0, len(DList)):
    tDL = tDL + DList[DL]
for BL in range(0, len(BList)):
    tBL = tBL + BList[BL]

values = list()
values.append(tPL)
values.append(tBL)
values.append(tDL)
values = [*values, *lengthBins, *widthBins, *circularEDBins, *circularityBins, *chordsBins]
print(values)

st.write('Writing pred_classes/pred_boxes output')
st.write(outputs["instances"].pred_classes)
st.write(outputs["instances"].pred_boxes)

st.write('Using Vizualizer to draw the predictions on Image')
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
st.image(out.get_image()[:, :, ::-1])
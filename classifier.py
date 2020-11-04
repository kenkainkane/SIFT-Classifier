import os
import sys
import cv2
import numpy as np

PATH = './img/train'  # path to train image dir
data_dir_list = os.listdir(PATH)

test_img = cv2.imread(sys.argv[1], 0)

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img_test, None)

bf = cv2.BFMatcher()

avgMatches = []

for dataset in data_dir_list:
  img_list = os.listdir(PATH + '/' + dataset + '/' + img, 0)
  # print('Loaded the images of dataset -', dataset')
  
  good = 0
  for img in img_list:
    train_img = cv2.imread(PATH + '/' + dataset + '/' + img, 0)
    kp2, des2 = sift.detectAndCompute(train_img, None)
    matches = bf.knnMatch(des1, des2, k=2)
    for m,n in matches:
            if m.distance < 0.75 * n.distance:
                good += 1
    avgMatches.append(good/len(img_list)/len(kp1))

# print(avgMatches)
print('Predict:', data_dir_list[np.argmax(avgMatches)])

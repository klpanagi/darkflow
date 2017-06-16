#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from distutils.dir_util import copy_tree
from shutil import copyfile
import sys
import os

datasetDir = sys.argv[1]
positiveDir = os.path.join(datasetDir, 'positive')
negativeDir = os.path.join(datasetDir, 'negative')
positiveN = int(sys.argv[2]) if len(sys.argv) >= 3 else 100000
negativeN = int(sys.argv[3]) if len(sys.argv) >= 4 else 100000

if not os.path.exists(datasetDir):
    print("Dataset directory does not exist: {0} ".format(datasetDir))
    sys.exit()
elif not os.path.exists(positiveDir):
    print("Dataset class directory does not exist: {0} ".format(positiveDir))
    sys.exit()
elif not os.path.exists(negativeDir):
    print(
        "Dataset negative directory does not exist: {0} ".format(negativeDir))
    sys.exit()
elif not os.path.exists(os.path.join(positiveDir, 'images')):
    print("Dataset class images directory does not exist: {0} ".format(
        os.path.join(positiveDir, 'images')))
    sys.exit()
elif not os.path.exists(os.path.join(positiveDir, 'annotations')):
    print("Dataset class annotations directory does not exist: {0} ".format(
        os.path.join(positiveDir, 'annotations')))
    sys.exit()
elif not os.path.exists(os.path.join(negativeDir, 'images')):
    print("Dataset negative images directory does not exist: {0} ".format(
        os.path.join(negativeDir, 'images')))
    sys.exit()
elif not os.path.exists(os.path.join(negativeDir, 'annotations')):
    print("Dataset negative annotations directory does not exist: {0} ".format(
        os.path.join(negativeDir, 'annotations')))
    sys.exit()

annotDir = os.path.join(datasetDir, 'Annotations')
imagesDir = os.path.join(datasetDir, 'Images')

if not os.path.exists(annotDir):
    os.makedirs(annotDir)
if not os.path.exists(imagesDir):
    os.makedirs(imagesDir)

for f in os.scandir(annotDir):
    os.unlink(f.path)
for f in os.scandir(imagesDir):
    os.unlink(f.path)

annotsNeg = [
    f for f in os.scandir(os.path.join(negativeDir, 'annotations'))
    if f.name.split('.')[1] == 'xml'
]
annotsPos = [
    f for f in os.scandir(os.path.join(positiveDir, 'annotations'))
    if f.name.split('.')[1] == 'xml'
]

imagesNeg = [
    f for f in os.scandir(os.path.join(negativeDir, 'images'))
    if f.name.split('.')[1] in ('png', 'jpg')
]
imagesPos = [
    f for f in os.scandir(os.path.join(positiveDir, 'images'))
    if f.name.split('.')[1] in ('png', 'jpg')
]

if len(imagesPos) != len(annotsPos):
    print(
        "Number of negative images and annotation files (xml) are not equal!!")
    sys.exit()
elif len(imagesNeg) != len(annotsNeg):
    print(
        "Number of positive images and annotation files (xml) are not equal!!")
    sys.exit()

positiveN = positiveN if positiveN < len(imagesPos) else len(imagesPos)
negativeN = negativeN if negativeN < len(imagesNeg) else len(imagesNeg)


for i in range(0, positiveN):
    copyfile(annotsPos[i].path, os.path.join(annotDir, annotsPos[i].name))
    copyfile(imagesPos[i].path, os.path.join(imagesDir, imagesPos[i].name))
for i in range(0, negativeN):
    copyfile(annotsNeg[i].path, os.path.join(annotDir, annotsNeg[i].name))
    copyfile(imagesNeg[i].path, os.path.join(imagesDir, imagesNeg[i].name))

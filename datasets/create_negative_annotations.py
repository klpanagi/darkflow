#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import os

negativeDir = sys.argv[1]
className = "negative"
annotationsDir = os.path.join(negativeDir, '..', 'annotations')

annotationFile = open('template_annot.xml').read()
negativeImages = [
    f for f in os.listdir(negativeDir)
    if os.path.isfile(os.path.join(negativeDir, f)) and f.split('.')[1] in (
        'jpg', 'png')
]

if not os.path.exists(annotationsDir):
    os.makedirs(annotationsDir)

counter = 0
for i, fimg in enumerate(negativeImages):
    name = fimg.split('.')[0]
    format = fimg.split('.')[0]
    with open(os.path.join(annotationsDir, name + '.xml'), 'w') as fstream:
        f = annotationFile.replace("FILENAME", name).replace(
            "CLASS", className)
        fstream.write(f)
    counter += 1

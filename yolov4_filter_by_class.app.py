import sys
import numpy as np
import tensorflow as tf
from core.yolov4 from YOLOv4, decode
from core.utils from load_weights


classes = []
names_file = ''
names = []


def on_set(k, v):
    if k == 'classes':
        global classes
        classes = v.split(',')
    elif k == 'names_file':
        global names_file
        global names
        names_file = ''
        names = util.read_class_names(names_file)


def on_get(k):
    if k == 'classes':
        return ','.join(classes)
    elif k == 'names_file':
        return names_file


def on_run(bboxes):

    filterd_bboxes = []
    for b in bboxes:
        n = names[b[5]]
        if n in classes:
            filterd_bboxes.append(b)

    # bboxes[[xmin, ymin, xmax, ymax, score, class]]
    # sys.stdout.write(f"[yolov4 filter by class] bboxes {bboxes}")
    # sys.stdout.flush()

    return {
        'filterd_bboxes' : filterd_bboxes
    }


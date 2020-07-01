import sys
import numpy as np
import tensorflow as tf
from core.yolov4 import YOLOv4, decode
from core import utils

model = None

weights = ''
input_size = 608
num_classes = 80
conf_threshold = 0.5
iou_threshold = 0.45


ANCHOR_DEFAULT = '12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401'
anchors = None

STRIDES = [8, 16, 32]
XYSCALE = [1.2, 1.1, 1.05]


def on_set(k, v):
    if k == 'weights':
        global weights
        weights = v
    elif k == 'input_size':
        global input_size
        input_size = int(v)
    elif k == 'num_classes':
        global num_classes
        num_classes = int(v)
    elif k == 'conf_threshold':
        global conf_threshold
        conf_threshold = float(v)
    elif k == 'iou_threshold':
        global iou_threshold
        iou_threshold = float(v)


def on_get(k):
    if k == 'weights':
        return weights
    elif k == 'input_size':
        return str(input_size)
    elif k == 'num_classes':
        return str(num_classes)
    elif k == 'conf_threshold':
        return str(conf_threshold)
    elif k == 'iou_threshold':
        return str(iou_threshold)


def on_init():
    global model
    global anchors

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            sys.stdout.write(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs\n")
            sys.stdout.flush()
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            sys.stdout.write(e)
            sys.stdout.flush()

    input_layer = tf.keras.layers.Input([input_size, input_size, 3])
    feature_maps = YOLOv4(input_layer, num_classes)
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, num_classes, i)
        bbox_tensors.append(bbox_tensor)
    model = tf.keras.Model(input_layer, bbox_tensors)

    if weights.split(".")[len(weights.split(".")) - 1] == "weights":
        utils.load_weights(model, weights)
    else:
        model.load_weights(weights).expect_partial()

    anchors = utils.get_anchors(ANCHOR_DEFAULT)
    return True


def on_run(image):
    image_data = utils.image_preporcess(
        np.copy(image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    pred_bbox = model.predict(image_data)

    pred_bbox = utils.postprocess_bbbox(pred_bbox, anchors, STRIDES, XYSCALE)
    bboxes = utils.postprocess_boxes(pred_bbox, image.shape[:-1], input_size, conf_threshold)
    bboxes = utils.nms(bboxes, iou_threshold, method='nms')

    # bboxes[[xmin, ymin, xmax, ymax, score, class]]
    sys.stdout.write(f"[yolov4 detect] bboxes {bboxes}")
    sys.stdout.flush()

    return {
        'bboxes' : np.array(bboxes)
    }


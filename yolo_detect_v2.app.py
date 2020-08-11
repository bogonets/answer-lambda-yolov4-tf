import sys
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
# from core.yolov4 import YOLOv4, decode
# from core import utils
import cv2

model_name = ''
model = None

weights = ''
input_size = 608
num_classes = 80
conf_threshold = 0.5
iou_threshold = 0.45
gpu = 0
gpu_memory_limit = 1024

ANCHOR_V4_DEFAULT = '12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401'
ANCHOR_V3_DEFAULT = '10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326'
anchors = None

STRIDES = [8, 16, 32]
XYSCALE = [1.2, 1.1, 1.05]

saved_model_loaded = None
infer = None


def on_set(k, v):
    if k == 'model':
        global model_name
        model_name = v
    elif k == 'weights':
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
    elif k == 'gpu':
        global gpu
        gpu = int(v)
    elif k == 'gpu_memory_limit':
        global gpu_memory_limit
        gpu_memory_limit = int(v)


def on_get(k):
    if k == 'model':
        return model_name
    elif k == 'weights':
        return weights
    elif k == 'input_size':
        return str(input_size)
    elif k == 'num_classes':
        return str(num_classes)
    elif k == 'conf_threshold':
        return str(conf_threshold)
    elif k == 'iou_threshold':
        return str(iou_threshold)
    elif k == 'gpu':
        return str(gpu)
    elif k == 'gpu_memory_limit':
        return str(gpu_memory_limit)


def on_init():
    global model
    global anchors
    global saved_model_loaded
    global infer

    gpus = tf.config.experimental.list_physical_devices('GPU')
    sys.stderr.write(f"[yolo_detect_v2.on_init] gpus({len(gpus)}) : {gpus}\n")
    sys.stderr.write(f"[yolo_detect_v2.on_init] selected gpu({gpu}) : {gpus[gpu]}\n")
    sys.stderr.flush()
    if gpus:
        # Restrict TensorFlow to only allocate some of memory on the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')
            # tf.config.experimental.set_memory_growth(gpus[gpu], True)
            tf.config.experimental.set_virtual_device_configuration(
                gpus[gpu],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            sys.stderr.write(f"[yolo_detect_v2.on_init] {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs\n")
            sys.stderr.flush()
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            sys.stderr.write(f'[yolo_detect.on_init] RuntimeError - {e}')
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f'[yolo_detect.on_init] Error - {e}')
            sys.stderr.flush()
    sys.stderr.write(f"[yolo_detect_v2.on_init] weights - {weights}\n")
    sys.stderr.flush()
    saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
    sys.stderr.write(f"[yolo_detect_v2.on_init] saved_model_loaded - {saved_model_loaded}\n")
    sys.stderr.flush()
    infer = saved_model_loaded.signatures['serving_default']
    sys.stderr.write(f"[yolo_detect_v2.on_init] infer- {infer}\n")
    sys.stderr.flush()

    #        return False

    # try:
    #     # Specify an invalid GPU device
    #     #with tf.device(f'/device:GPU:{gpu}'):

    #     sys.stderr.write(f"[yolo_detect_v2.on_init] weights - {weights}\n")
    #     sys.stderr.flush()
    #     saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
    #     sys.stderr.write(f"[yolo_detect_v2.on_init] saved_model_loaded - {saved_model_loaded}\n")
    #     sys.stderr.flush()
    #     infer = saved_model_loaded.signatures['serving_default']
    #     sys.stderr.write(f"[yolo_detect_v2.on_init] infer- {infer}\n")
    #     sys.stderr.flush()

    # except RuntimeError as e:
    #     sys.stderr.write(f'[yolo_detect.on_init] RuntimeError - {e}')
    #     sys.stderr.flush()
    # except Exception as e:
    #     sys.stderr.write(f'[yolo_detect.on_init] Error - {e}')
    #     sys.stderr.flush()

    return True


def on_run(image):
    # try:
    # Specify an invalid GPU device
    # with tf.device(f'/device:GPU:{gpu}'):
    # sys.stderr.write(f'[yolo_detect.on_run] image.shape - {image.shape}\n')
    # sys.stderr.flush()
    image_data = cv2.resize(image, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    batch_data = tf.constant(image_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]
    # sys.stderr.write(f'[yolo_detect.on_run] pred_bbox - {pred_bbox}\n')
    # sys.stderr.flush()

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou_threshold,
        score_threshold=conf_threshold
    )
    # bboxes = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    # boxes = [[[xmin, ymin, xmax, ymax], [...]]]
    # scores = [[1,2]] -> [[1],[2]]
    # classes = [[0,0]]
    # sys.stderr.write(f'[yolo_detect.on_run] boxes - {boxes}\n')
    # sys.stderr.write(f'[yolo_detect.on_run] scores - {scores}\n')
    # sys.stderr.write(f'[yolo_detect.on_run] classes - {classes}\n')
    # sys.stderr.write(f'[yolo_detect.on_run] valid_detections - {valid_detections}\n')
    # sys.stderr.flush()
    bboxes = boxes.numpy()[0]
    bboxes = np.append(bboxes, scores.numpy().reshape(-1,1), axis=1)
    bboxes = np.append(bboxes, classes.numpy().reshape(-1,1), axis=1)

    # bboxes[[xmin, ymin, xmax, ymax, score, class]]
    # sys.stderr.write(f'[yolo_detect.on_run] bboxes - {bboxes}\n')
    # sys.stderr.flush()

    return {
            'bboxes' : np.array(bboxes)
        }
    # except RuntimeError as e:
    #     sys.stderr.write(f'[yolo_detect.on_run] RuntimeError - {e}')
    #     sys.stderr.flush()
    # except Exception as e:
    #     sys.stderr.write(f'[yolo_detect.on_run] Error - {e}')
    #     sys.stderr.flush()

    # return {}



import sys
import numpy as np
import tensorflow as tf
from core.yolov4 import YOLOv4, decode
from core import utils


gpu = 0


def on_set(k, v):
    if k == 'gpu':
        global gpu
        gpu = int(v)


def on_get(k):
    if k == 'gpu':
        return str(gpu)


def on_init():

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) <= gpu:
        sys.stderr.write("wrong index")
        sys.stderr.flush()
        return True
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[gpu],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        sys.stdout.write(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs\n")
        sys.stdout.flush()
    except Exception as e:
        # Virtual devices must be set before GPUs have been initialized
        sys.stdout.write(f'{e}')
        sys.stdout.flush()

    return True


def on_run():
    pass


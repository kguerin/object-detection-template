from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
sys.path.append("slim")

from support import draw_outline, draw_rect, draw_text, bb_hw, show_img, open_image

import tensorflow as tf
import numpy as np
import pandas as pd

from PIL import Image
from collections import namedtuple, OrderedDict
from object_detection.utils import dataset_util
from tqdm import tqdm

import path
import cv2
import os
import io
import numpy as np

# Install extra-dependency
# pip -q install pycocotools
import functools
import json
import tensorflow as tf

from object_detection import trainer
from object_detection.builders import dataset_builder
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import dataset_util

from support import get_train_config, get_eval_config

# For reproducibility
from tensorflow import set_random_seed
from numpy.random import seed
seed(2018)
set_random_seed(2018)

dataSetPath = "/floyd/input/data/"

labelfilepath = "dynamiclabels.txt"

# Path to the CSV input
CSV_INPUT = dataSetPath + "annotations.csv"
# Path to the image directory
IMAGE_DIR = dataSetPath + "images/"

# Path to output TFRecord
OUTPUT = '/floyd/home/tfrecords_data'
data = pd.read_csv(CSV_INPUT)
recordCount = data.shape[0]
#  Train / Val split
# eval is 10% of the dataset the rest is for training
spl = int(recordCount - (recordCount/10))
train_df = data.iloc[:spl]
eval_df = data.iloc[spl:]

# dynamic object labels
objectlabels = []
for index, row in data.iterrows():
    if row["class"] not in objectlabels:
        objectlabels.append(row["class"])

def writeLabelsTxtFile(objectlabels):
    content = ""
    for index, label in enumerate(objectlabels):
        thiscontent = "item {{\n id: {0}\n name: '{1}' \n}} \n\n".format(index+1,label)
        content = content + thiscontent
        
    print(content)
    f= open(labelfilepath,"w+")
    f.write(content)
#write labels file to reference    

writeLabelsTxtFile(objectlabels)    
print("Creating tensorflowdata for {}".format(objectlabels))

#set number of epochs to 10000 for each label
NUM_EPOCHS  = 10000 * len(objectlabels)

def writePipelineConfig(objectlabels):
    template = open("models/ssdlite_mobilenet_v2_coco_2018_05_09/pipeline-template.config","r")
    templateContents = template.read()
    numberOfLabels = len(objectlabels)
    epochs = len(objectlabels) * 10000
    withnumclasses = templateContents.replace("#num_classes#", str(numberOfLabels), 1)
    finalContent = withnumclasses.replace("#num_steps#", str(epochs), 1)
    newConfigFile = open("models/ssdlite_mobilenet_v2_coco_2018_05_09/pipeline.config","w+") 
    newConfigFile.write(finalContent)
    newConfigFile.close()
    print("training pipeline configured")
    
writePipelineConfig(objectlabels)

#class labels are dynamic
def class_text_to_int(row_label):
    retvalu = objectlabels.index(row_label) + 1
    """Replace the label with an int"""
    if retvalu > 0: 
        return retvalu
    else:
        None
        
def split(df, group):
    """For each images, return a data object with all the labels/bbox in the images
    
    e.g.
    
    [data(filename='1.jpg', object=  filename  width  height  class  xmin  ymin  xmax  ymax
     0    1.jpg   2048    1251  syd   706   513   743   562),
     data(filename='10.jpg', object=   filename  width  height  class  xmin  ymin  xmax  ymax
     1    10.jpg   1600     980  syd   715   157   733   181
     19   10.jpg   1600     980  syd   428    83   483   145),
     ...
     data(filename='9.jpg', object=   filename  width  height  class  xmin  ymin  xmax  ymax
     17    9.jpg   1298     951  syd   231   735   261   769)]
     
    """
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path):
    """
    From data group object to TFRecord file
    
    Note: we are handling JPG data format and bbox labels. 
    If you need to work on PNG data with mask or polygon labels, you will have to edit the code a bit.
    """
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes)
    }))
    return tf_example

##################################################
# create tf records #
##################################################

# Prepare the TFRecord Output
TRAIN_RECORD_FILE = os.path.join(OUTPUT, 'train.tfrecord')
EVAL_RECORD_FILE = os.path.join(OUTPUT, 'eval.tfrecord')

# TRAIN TFRecord
writer = tf.python_io.TFRecordWriter(path=TRAIN_RECORD_FILE)
path_to_images = os.path.join(os.getcwd(), IMAGE_DIR)

# From CSV to TFRecord
grouped = split(train_df, 'filename')
for group in tqdm(grouped):
    tf_example = create_tf_example(group, path_to_images)
    writer.write(tf_example.SerializeToString())
writer.close()

# EVAL TFRecord
writer = tf.python_io.TFRecordWriter(path=EVAL_RECORD_FILE)
path_to_images = os.path.join(os.getcwd(), IMAGE_DIR)

# From CSV to TFRecord
grouped = split(eval_df, 'filename')
for group in tqdm(grouped):
    tf_example = create_tf_example(group, path_to_images)
    writer.write(tf_example.SerializeToString())
writer.close()

print('Successfully created the TFRecords: {}'.format(OUTPUT))



##################################################
# TRAINING CONFIGURATION (Local, Multi-GPUs, Distributed) #
##################################################

MASTER = ''  # Name of the TensorFlow master to use
TASK = 0  # task id
NUM_CLONES = 1  # Number of clones to deploy per worker.

# Force clones to be deployed on CPU.  Note that even if 
# set to False (allowing ops to run on gpu), some ops may 
# still be run on the CPU if they have no GPU kernel.
CLONE_ON_CPU = False

WORKER_REPLICAS = 1  # Number of worker+trainer replicas.
PS_TASKS = 0  # Number of parameter server tasks. If None, does not use a parameter server.
TRAIN_DIR = 'trained_models/ssdlite_mobilenet_v2_coco_{}'.format(baseModelName)  # Directory to save the checkpoints and training summaries.

# Path to a pipeline_pb2.TrainEvalPipelineConfig config
# file. If provided, other configs are ignored
PIPELINE_CONFING_PATH = 'models/ssdlite_mobilenet_v2_coco_2018_05_09/pipeline.config'
TRAIN_CONFIG_PATH = ''  # Path to a train_pb2.TrainConfig config file.
INPUT_CONFIG_PATH = ''  # Path to an input_reader_pb2.InputReader config file.
MODEL_CONFIG_PATH = ''  # Path to a model_pb2.DetectionModel config file.

# Get configuration for training
(create_input_dict_fn,
model_fn,
train_config,
master,
task,
worker_job_name,
ps_tasks,
worker_replicas,
is_chief,
graph_rewriter_fn) = get_train_config(TASK,
                                     PS_TASKS,
                                     TRAIN_DIR,
                                     PIPELINE_CONFING_PATH,
                                     TRAIN_CONFIG_PATH,
                                     MODEL_CONFIG_PATH,
                                     INPUT_CONFIG_PATH,
                                     WORKER_REPLICAS,
                                     MASTER)

train_config.num_steps = NUM_EPOCHS

##################################################
# TRAIN THE MODEL #
##################################################

trainer.train(
  create_input_dict_fn,
  model_fn,
  train_config,
  master,
  task,
  NUM_CLONES,
  worker_replicas,
  CLONE_ON_CPU,
  ps_tasks,
  worker_job_name,
  is_chief,
  TRAIN_DIR,
  graph_hook_fn=graph_rewriter_fn)


##################################################
# EVALUATE THE MODEL #
##################################################

from object_detection import evaluator
from object_detection.utils import label_map_util

tf.reset_default_graph()

##############################
# CONFIGURATION (Evaluation) #
##############################

# Directory to write eval summaries to.
EVAL_DIR = 'trained_models/ssdlite_mobilenet_v2_coco_2018_05_09/eval' 

# If training data should be evaluated for this job.
EVAL_TRAINING_DATA = False

# Option to only run a single pass of evaluation. 
# Overrides the `max_evals` parameter in the provided config
RUN_ONCE = True

# Directory containing checkpoints to evaluate, typically
# set to `train_dir` used in the training job.
CHECKPOINT_DIR = TRAIN_DIR

EVAL_CONFIG_PATH = ''  # Path to a eval_pb2.TrainConfig config file.
EVAL_INPUT_CONFIG_PATH = ''  # Path to an input_reader_pb2.InputReader config file.
MODEL_CONFIG_PATH = ''  # Path to a model_pb2.DetectionModel config file.

# Get configuration for Evaluation
(create_input_dict_fn,
model_fn,
eval_config,
categories,
graph_rewriter_fn) = get_eval_config(EVAL_DIR,
                                    PIPELINE_CONFING_PATH,
                                    EVAL_CONFIG_PATH,
                                    MODEL_CONFIG_PATH,
                                    EVAL_INPUT_CONFIG_PATH,
                                    EVAL_TRAINING_DATA,
                                    RUN_ONCE)

######################################
# HOW TO EDIT EVAL CONFIG FROM CODE #
######################################

# You can change the setting in this way,
# e.g. for evaluate only once
# eval_config.max_evals = 1

print("Eval configuration: \n", '-'*30, '\n', eval_config)

######################################
# EXPORT THE MODEL #
######################################

# Exporting the model for Evaluation
from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2

slim = tf.contrib.slim
tf.reset_default_graph()

config_override=''
input_shape=None
trained_checkpoint_prefix = os.path.join(TRAIN_DIR, 'model.ckpt-' + str(NUM_EPOCHS) ) # EDIT: model.ckpt-<iteration>

output_directory='trained_models/exported_model'

# Type of input node. Can be one of :
# [`image_tensor`, `encoded_image_string_tensor`,`tf_example`]
input_type='image_tensor' 

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.gfile.GFile(PIPELINE_CONFING_PATH, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)
text_format.Merge(config_override, pipeline_config)
if input_shape:
    input_shape = [
        int(dim) if dim != '-1' else None
        for dim in input_shape.split(',')
    ]
else:
    input_shape = None
exporter.export_inference_graph(input_type, pipeline_config,
                                  trained_checkpoint_prefix,
                                  output_directory, input_shape)
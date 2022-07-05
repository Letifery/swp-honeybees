from utils.dataloader import DataLoader
from utils.logger import Logger
from utils.preprocessing import Preprocessing
from models.cnn3d import ConvNet3D

from copy import deepcopy
import tensorflow as tf
import time
import numpy as np
import sys, gc

def y_to_numbers(y,all_5_classes=False):
    label_to_number={
        'waggle':0,
        'trembling':4 if all_5_classes else 3,
        'ventilating':2,
        'other':3,
        'activating':1,
    }
    nb_classes = 5 if all_5_classes else 4
    y = np.vectorize(label_to_number.get)(y)

    return np.array(np.eye(nb_classes)[np.array(y).reshape(-1)]),label_to_number

def aggregate_data(data, pickle_file):
    t = time.time()
    images, json_files, paths = np.array(list(zip(*data)))
    angles, classes = np.array([]), np.array([])

    for dic in json_files:
        angles = np.append(angles, dic["waggle_angle"])
        classes = np.append(classes, pickle_file[dic["waggle_id"]][0])
    print("[AGGREGATE] <DATA,PICKLE> time needed: %s seconds\n" % (time.time()-t))
    return (images, (angles, classes), json_files, paths)
    
#setup
path_data = r"I:\tmp_swp\wdd_ground_truth"
path_pickle = r"I:\tmp_swp\ground_truth_wdd_angles.pickle"
path_testsuite = r"I:\tmp_swp\data\pptestsuite.json"
#Sets the filename prefix for the logging tool
model_name = "cnn3D-mv2"

dl = DataLoader()
data_logger = Logger("review_%s.log" % model_name)
#Below should be uncommented for the first iteration
data_logger.log_data([["ID", "CPU", "runtime", "loss", "cat_accuracy", "cat_entropy", "pp_layers", "confusion_matrix"]])
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.85)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

data = list(dl.get_data([path_data]))
pickle_file = list(dl.get_pickle([path_pickle]))[0]


X, Y, json_files, paths = aggregate_data(data, pickle_file)
Y_classes, Y_angles = Y[1], Y[0]
y_one_hot, _ = y_to_numbers(Y_classes)

ID_START = 0

hyper_X = list(dl.get_json([path_testsuite]))[0]["testsuiteSlice"]

###############################
#gc.set_debug(gc.DEBUG_STATS)
###############################
    
for i in range(len(hyper_X)):
    pp = Preprocessing(deepcopy(X))
    Xpp = pp.preprocess_data(hyper_X[i])
    summary_logger = Logger("%s-%s.log" % (model_name,(i+ID_START)))
    
    OOM_interrupt = 0
    t = time.time()
    
    cmodel = ConvNet3D()
    model = cmodel.setup_model(Xpp)
    try:
        results, summary_string = cmodel.evaluate_model(model, Xpp, y_one_hot)
    except tf.errors.ResourceExhaustedError:
        with tf.device('/cpu:0'):
            t = time.time()
            print("\033[93m[WARNING] Could not use CUDA for testset iteration %s, will switch to CPU instead\033[0m" % i)
            OOM_interrupt = 1
            results, summary_string = cmodel.evaluate_model(model, Xpp, y_one_hot)
    data_logger.log_data([[(i+ID_START), OOM_interrupt, time.time()-t, *results[0], hyper_X[i], results[1]]])
    summary_logger.log_data([[summary_string]], "summaries")

    del cmodel, pp
    gc.collect()

from utils.dataloader import DataLoader
from utils.logger import Logger
from utils.preprocessing import Preprocessing
from models.cnn2D import ConvNet2D
from models.cnn2d_independent_of_clip_length import ConvNet2D_independent
from copy import deepcopy
from traceback import format_exc


import tensorflow as tf
import time
import numpy as np
import os, sys, gc
import platform
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

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
MODEL_NAME = "cnn2d_independent"

PATH_DATA = r"I:\tmp_swp\wdd_ground_truth"
PATH_PICKLE = r"I:\tmp_swp\ground_truth_wdd_angles.pickle"
PATH_TESTSUITE = r"I:\tmp_swp\data\pptestsuite.json"
PATH_SAVE_MODEL = "I:\tmp\MODEL_NAME.hdf5"


dl = DataLoader()
data_logger = Logger("review_%s.log" % MODEL_NAME)

sl = "/" if platform.system() == "Linux" else "\\."[0]

if not os.path.exists("logs%sdatalogs_%s%sreview_%s.log" % (sl, MODEL_NAME, sl, MODEL_NAME)):
    id_start = 0
    data_logger.log_data([["ID", "CPU", "t_overall", "loss", "cat_accuracy", "cat_entropy", "t_setup",
                    "t_pp", "t_model", "t_evmodel", "pp_layers", "confusion_matrix"]], "datalogs_%s" % MODEL_NAME)
else:
    df = pd.read_csv("logs%sdatalogs_%s%sreview_%s.log" % (sl, MODEL_NAME, sl, MODEL_NAME), on_bad_lines='skip')
    id_start = len(df.index)

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.85)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

data = list(dl.get_data([PATH_DATA]))
pickle_file = list(dl.get_pickle([PATH_PICKLE]))[0]

label_to_number={
        'waggle':0,
        'activating':1,
        'ventilating':2,
        'other':3,
        'trembling':2,
}

X, Y, json_files, paths = aggregate_data(data, pickle_file)
Y_classes, Y_angles = Y[1], Y[0]


Y = np.array([label_to_number[y] for y in Y_classes])
hyper_X = list(dl.get_json([PATH_TESTSUITE]))[0]["testsuiteA"]

for i in range(len(hyper_X)):
    OOM_interrupt = 0
    runtimes = [0]*4

    try:
        t = time.time()
        Xtmp = deepcopy(X)
        cmodel = ConvNet2D()
        pp = Preprocessing(Xtmp)
        summary_logger = Logger("summary_%s_%s.log" % ((i+id_start), MODEL_NAME))
        runtimes[0] = time.time()-t

        X_pp = pp.preprocess_data(hyper_X[i])
        runtimes[1] = time.time()-(t+sum(runtimes))

        model = cmodel.setup_model(X_pp)
        runtimes[2] = time.time()-(t+sum(runtimes))

        results, summary_string = cmodel.evaluate_model(model, X_pp, Y, PATH_SAVE_MODEL)
        runtimes[3] = time.time()-(t+sum(runtimes))

    except tf.errors.ResourceExhaustedError:
        try:
            with tf.device('/cpu:0'):
                print("\033[93m[WARNING] Could not use CUDA for testset iteration %s, will switch to CPU instead\033[0m" % i)
                OOM_interrupt = 1
                t = time.time()
                results, summary_string = cmodel.evaluate_model(model, X_pp, Y, PATH_SAVE_MODEL)
                runtimes[3] = time.time()-t
        except Exception:
            data_logger.log_data([[(i+id_start)]+["ERROR"]*11], "datalogs_%s" % MODEL_NAME)
            summary_logger.log_data([[format_exc()]], "datalogs_%s" % MODEL_NAME)
            continue
    except Exception:
        data_logger.log_data([[(i+id_start)]+["ERROR"]*11], "datalogs_%s" % MODEL_NAME)
        summary_logger.log_data([[format_exc()]], "datalogs_%s" % MODEL_NAME)
        continue

    data_logger.log_data([[(i+id_start), OOM_interrupt, sum(runtimes), *results[0], *runtimes,
                            str(hyper_X[i]), str(results[1])]], "datalogs_%s" % MODEL_NAME)
    summary_logger.log_data([[summary_string]], "datalogs_%s" % MODEL_NAME)
    del cmodel, pp, summary_logger
    gc.collect()

import random
import numpy as np
import os
from numpy.random import seed
import tensorflow as tf
random_seed = 1388   
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ['TF_CUDNN_DETERMINISTIC']='1'
# os.environ['PYTHONHASHSEED'] = str(random_seed)
# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)
seed(random_seed) 
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
tf.keras.utils.set_random_seed(random_seed)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
import datetime
import h5py
import json
import pandas as pd   
import sys
from scipy.io import savemat
from scipy.interpolate import interp1d
from scipy import optimize
from scipy.stats import gaussian_kde,pearsonr
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, ScalarFormatter
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib import cm, colors,rcParams
from matplotlib.font_manager import FontProperties
from statistics import mean
from sklearn.metrics import explained_variance_score,r2_score,median_absolute_error,mean_squared_error,mean_absolute_error
from sklearn.linear_model import LinearRegression
import keras.backend as K
from keras import layers, models, callbacks
from keras.models import Model,Sequential,load_model   
from keras.layers import Conv1D,Dense,Multiply,Input ,Reshape,Dot,Flatten,Lambda
from keras.callbacks import ModelCheckpoint , Callback
from keras.initializers import RandomNormal
plt.rcParams['font.sans-serif']=['Times New Roman']   
plt.rcParams["axes.unicode_minus"]=False  
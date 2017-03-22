#coding=utf-8
from matplotlib import pyplot as plt
import math
import numpy as np
import os
import pandas as pd
import random
import time
import cntk as C
import cntk.axis
from cntk.layers import Input, Dense, Dropout, Recurrence
from cntk.ops.functions import load_model

def generate_data(path, file, time_steps, NORMALIZE=1.0, val_size=0.1, ifNormalize=False):
    cache_path = os.path.join(path)
    cache_file = os.path.join(cache_path, file)
    try:
        data = pd.read_csv(cache_file, dtype=np.float32)
    except(e):
        print("Fail to read file.")
    if ifNormalize:
        data /= NORMALIZE

    sequence_length = time_steps + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(np.array(data[index: index + sequence_length]))
    result = np.array(result)

    result_x = {"test": []}
    result_y = {"test": []}    
    result_x["test"] = result[:, :-1]
    result_y["test"] = result[:, -1]

    # make result_y a numpy array
    result_y["test"] = np.array(result_y["test"])

    return result_x, result_y

def next_batch(x, y, ds):
    """get the next batch"""
    def as_batch(data, start, count):
        return data[start:start + count]

    for i in range(0, len(x[ds]), BATCH_SIZE):
        yield as_batch(X[ds], i, BATCH_SIZE), as_batch(Y[ds], i, BATCH_SIZE)

#Main run from here
TIMESTEPS = 64
BATCH_SIZE = TIMESTEPS * 1
Path = "D:\WriteTick"
DataFile = "RB9999In2014Ave500TicksWinNorm11-12.csv"
ModleFile = "LSTM-CNTK.dnn"

# to make things reproduceable, seed random
np.random.seed(0)

if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        C.device.set_default_device(C.device.cpu())
    else:
        C.device.set_default_device(C.device.gpu(0))

#Load data
print("Generating data")
X, Y = generate_data(Path, DataFile, TIMESTEPS, ifNormalize = False)

# Load the model
print("Loading model from " + Path + ModleFile)
z = load_model(Path + ModleFile)

# Show picture of testing
results = []
ds = "test"
for x_batch, _ in next_batch(X, Y, ds):
    pred = z.eval({z.arguments[0]: x_batch})
    results.extend(pred[:, 0])
fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111)
#Y[ds] = np.array([float(c[0]) * NORMALIZE for c in Y[ds]])
Y[ds] = np.array([float(c[0]) for c in Y[ds]])
ax.plot(Y[ds], label="True Data");
#results = np.array([c[0] * NORMALIZE for c in results])
results = np.array([c[0] for c in results])
ax.plot(results, label="Prediction");
ax.legend();
plt.show();
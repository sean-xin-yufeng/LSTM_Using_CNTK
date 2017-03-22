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

    # split the dataset into train, validatation and test sets on day boundaries
    tot_row = len(result)
    train_row = tot_row - int(tot_row * val_size)

    result_x = {"train": [], "val": []}
    result_y = {"train": [], "val": []}    

    train = result[:train_row, :]
    val = result[train_row:, :]
    np.random.shuffle(train)#only shuffled train
    result_x["train"] = train[:, :-1]
    result_y["train"] = train[:, -1]
    result_x["val"] = val[:, :-1]
    result_y["val"] = val[:, -1]

    # make result_y a numpy array
    for ds in ["train", "val"]:
        result_y[ds] = np.array(result_y[ds])

    return result_x, result_y

def next_batch(x, y, ds):
    """get the next batch"""
    def as_batch(data, start, count):
        return data[start:start + count]

    for i in range(0, len(x[ds]), BATCH_SIZE):
        yield as_batch(X[ds], i, BATCH_SIZE), as_batch(Y[ds], i, BATCH_SIZE)

def create_model(x):
    """Create the model for time series prediction"""
    with C.layers.default_options(initial_state = 0.1):
        m = C.layers.Recurrence(C.layers.LSTM(TIMESTEPS))(x)
        m = C.sequence.last(m)
        m = C.layers.Dropout(0.2)(m)
        m = cntk.layers.Dense(1)(m)
        return m

# validate
def get_mse(X,Y,labeltxt):
    result = 0.0
    for x1, y1 in next_batch(X, Y, labeltxt):
        eval_error = trainer.test_minibatch({x : x1, l : y1})
        result += eval_error
    return result/len(X[labeltxt])

#Main run from here
EPOCHS = 16
NORMALIZE = 0.01 #尝试在数据中预处理实现窗口正态化
TIMESTEPS = 16
BATCH_SIZE = TIMESTEPS * 1
Path = "D:\WriteTick"
DataFile = "RB9999In2014Ave500TicksWinNorm1-10.csv"
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

print("Creating model")
# input sequences
x = C.layers.Input(1)

# create the model
z = create_model(x)

# expected output (label), also the dynamic axes of the model output
# is specified as the model of the label input
l = C.layers.Input(1, dynamic_axes=z.dynamic_axes, name="y")

# the learning rate
learning_rate = 0.005
lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)

# loss function
loss = C.ops.squared_error(z, l)

# use squared error to determine error for now
error = C.ops.squared_error(z, l)

# use adam optimizer
momentum_time_constant = C.learner.momentum_as_time_constant_schedule(BATCH_SIZE / -math.log(0.9)) 
learner = C.learner.adam_sgd(z.parameters, lr = lr_schedule, momentum = momentum_time_constant)
trainer = C.Trainer(z, (loss, error), [learner])

# training
print("Training model")
loss_summary = []

start = time.time()
for epoch in range(1, EPOCHS+1):
    for x_batch, l_batch in next_batch(X, Y, "train"):
        trainer.train_minibatch({x: x_batch, l: l_batch})
    training_loss = C.utils.get_train_loss(trainer)
    loss_summary.append(training_loss)
    print("epoch: {}, loss: {:.4f}".format(epoch, training_loss))

print("Training took {:.1f} sec".format(time.time() - start))

# Print the train and validation errors
for labeltxt in ["train", "val"]:
    print("MSE for {}: {:.6f}".format(labeltxt, get_mse(X, Y, labeltxt)))

# Save the model
z.save_model(Path + ModleFile)
print("Model saved in " + Path + ModleFile)

# Show picture of validating
results = []
ds = "val"
for x_batch, _ in next_batch(X, Y, ds):
    pred = z.eval({x: x_batch})
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
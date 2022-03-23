import argparse
from datetime import datetime, timedelta
from tabnanny import verbose
import numpy as np
import tensorflow as tf
import os
import sys
import time
from sklearn import metrics
from shutil import rmtree
from joblib import dump

startTime = time.time() # Tracks how long script takes

# os.environ['CUDA_VISIBLE_DEVICES']="0"

from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR,'..', 'models')) 
sys.path.append(os.path.join(BASE_DIR,'..', 'utils')) 
import pct_tf2 as MODEL
import provider

# gpus = tf.config.experimental.list_physical_devices('GPU') 
# print(gpus)
# tf.config.experimental.set_memory_growth(gpus[0], True)
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# tf.debugging.set_log_device_placement(True)

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='pct_tf2', help='Model name [default: pct]')
parser.add_argument('--log_dir', default='bes_dev', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=50, help='Point Number  [default: 50]')
parser.add_argument('--num_bes', type=int, default=142, help='Number of BES variables  [default: 142]')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 200]')
parser.add_argument('--batch_size', type=int, default=100, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate [default: 0.001]')
parser.add_argument('--sub_epoch', type=int, default=100000, help='Sub Epoch Size [default: 100000]')

parser.add_argument('--data_dir', default='/uscms/home/bonillaj/nobackup/h5samples_ULv1/', help='directory with data [default: hdf5_data]')
# parser.add_argument('--nfeat', type=int, default=4, help='Number of features PF [default: 16]')
# parser.add_argument('--ncat', type=int, default=2, help='Number of categories [default: 2]')
parser.add_argument('--simple', action='store_true', default=False,help='Use simplified model')
parser.add_argument('--test', action='store_true', default=False,help='start a test training')



FLAGS = parser.parse_args()
DATA_DIR = FLAGS.data_dir
BATCH_SIZE = FLAGS.batch_size
# NUM_FEAT = FLAGS.nfeat
NUM_FEAT = 13, 9
# NUM_POINT = FLAGS.num_point
NUM_POINT = 50, 10
NUM_BES = FLAGS.num_bes
# NUM_CLASSES = FLAGS.ncat
NUM_CLASSES = 6
MAX_EPOCH = FLAGS.max_epoch
LEARNING_RATE = FLAGS.learning_rate

SUB_EPOCH_SIZE = FLAGS.sub_epoch

MODEL_FILE = os.path.join(BASE_DIR, '..', 'models',FLAGS.model+'.py')
LOG_DIR = os.path.join('../logs',FLAGS.log_dir)
print("\nLOG DIR: " + LOG_DIR + "\n")

if os.path.exists(LOG_DIR): rmtree(LOG_DIR)
os.makedirs(LOG_DIR)
# if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train_tf2.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


LEARNING_RATE_CLIP = 1e-6
EARLY_TOLERANCE=15

FRAMES, DEP_FRAMES = provider.getFrames(FLAGS.log_dir)
print(FRAMES)

simple = False
if "S" in FLAGS.log_dir: simple = True
# print(simple)

TRAIN_FILE = [os.path.join(DATA_DIR, mySamp+'Sample_2017_BESTinputs_train_flattened.h5') for mySamp in ["WW","ZZ","HH","TT","BB","QCD"]]
TEST_FILE  = [os.path.join(DATA_DIR, mySamp+'Sample_2017_BESTinputs_validation_flattened.h5') for mySamp in ["WW","ZZ","HH","TT","BB","QCD"]]

MASK=[]
if "BES_vars" in FRAMES:
    maskPath = "/uscms/home/msabbott/nobackup/general/CMSSW_10_6_27/src/abbottBEST/BEST/training/models/newBEST_maskFix/longBasic_ak8/fixBESTMask_ak8.txt"
    maskFile = open(maskPath, "r")
    maskIndex = []
    for line in maskFile:
        maskIndex.append(line.split(':')[0])
        # maskIndex.append(int(line.split(':')[0]))
    maskFile.close()
    # myMask = [True if str(ind) in maskIndex else False for ind in range(596)]
    MASK = [True if str(ind) in maskIndex else False for ind in range(551)]
    NUM_BES = np.array(MASK).sum()

# mask = maskIndex
# NUM_BES = len(maskIndex)

train_data = provider.load_bes(TRAIN_FILE, FRAMES, BATCH_SIZE, NUM_BES, is_training = True, mask=MASK)
test_data  = provider.load_bes(TEST_FILE,  FRAMES, BATCH_SIZE, NUM_BES, mask=MASK)

# TRAIN_FILE = os.path.join(DATA_DIR, 'HHSample_2017_BESTinputs_validation_flattened_standardized_tiny.h5')
# train_data = provider.load_bes(TRAIN_FILE,batch_size=BATCH_SIZE)
# TEST_FILE = os.path.join(DATA_DIR, 'HHSample_2017_BESTinputs_validation_flattened_standardized_tiny.h5')
# test_data = provider.load_bes(TEST_FILE,batch_size=BATCH_SIZE)


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def BEST_PCT():
    besFlag = False
    input_global = np.array([0])
    # if "BES_vars" in FRAMES: 
        # besFlag = True
        # input_global = Input((NUM_BES,))

    pointclouds = []
    preds = []
    for frame in FRAMES:
        if frame == "BES_vars": # BES_vars will ALWAYS be the last frame, from getFrames()
            besVars, besPred = MODEL.newBESTNN(NUM_BES, NUM_CLASSES)
            pointclouds.append(besVars)
            preds.append(besPred)
            continue
        # if FLAGS.simple:
        if simple:
            pclouds, pred = MODEL.PCT_simple(besFlag, input_global, NUM_POINT[0], NUM_FEAT[0], NUM_CLASSES)
            pointclouds.append(pclouds)
            preds.append(pred)
        else:
            pclouds, pred = MODEL.PCT(besFlag, input_global, NUM_POINT[0], NUM_FEAT[0], NUM_CLASSES)
            pointclouds.append(pclouds)
            preds.append(pred)

    net = tf.concat(preds,-1)
    net = Dense(256 ,activation='relu')(net)    
    net = Dropout(0.2)(net)
    net = Dense(128 ,activation='relu')(net)    
    net = Dropout(0.2)(net)
    outputs = Dense(NUM_CLASSES,activation='softmax')(net)

    if besFlag: return tuple([input_global] + pointclouds + [outputs])
    else:       return tuple(pointclouds + [outputs])
    """
    if FLAGS.simple:
        pointclouds_lab,pred_lab =MODEL.PCT_simple(input_global,NUM_POINT,NUM_FEAT,NUM_CLASSES, besFlag)
        pointclouds_higgs,pred_higgs =MODEL.PCT_simple(input_global,NUM_POINT,NUM_FEAT,NUM_CLASSES, besFlag)
        pointclouds_bottom,pred_bottom =MODEL.PCT_simple(input_global,NUM_POINT,NUM_FEAT,NUM_CLASSES, besFlag)
        pointclouds_top,pred_top =MODEL.PCT_simple(input_global,NUM_POINT,NUM_FEAT,NUM_CLASSES, besFlag)
        pointclouds_W,pred_W =MODEL.PCT_simple(input_global,NUM_POINT,NUM_FEAT,NUM_CLASSES, besFlag)
        pointclouds_Z,pred_Z =MODEL.PCT_simple(input_global,NUM_POINT,NUM_FEAT,NUM_CLASSES, besFlag)
    else:
        pointclouds_lab,pred_lab =MODEL.PCT(input_global,NUM_POINT,NUM_FEAT,NUM_CLASSES, besFlag)
        pointclouds_higgs,pred_higgs =MODEL.PCT(input_global,NUM_POINT,NUM_FEAT,NUM_CLASSES, besFlag)
        pointclouds_bottom,pred_bottom =MODEL.PCT(input_global,NUM_POINT,NUM_FEAT,NUM_CLASSES, besFlag)
        pointclouds_top,pred_top =MODEL.PCT(input_global,NUM_POINT,NUM_FEAT,NUM_CLASSES, besFlag)
        pointclouds_W,pred_W =MODEL.PCT(input_global,NUM_POINT,NUM_FEAT,NUM_CLASSES, besFlag)
        pointclouds_Z,pred_Z =MODEL.PCT(input_global,NUM_POINT,NUM_FEAT,NUM_CLASSES, besFlag)

    net = tf.concat([pred_lab,pred_higgs,pred_bottom,pred_top,pred_W,pred_Z],-1)
    net = Dense(256 ,activation='relu')(net)    
    net = Dropout(0.2)(net)
    net = Dense(128 ,activation='relu')(net)    
    net = Dropout(0.2)(net)
    outputs = Dense(NUM_CLASSES,activation='softmax')(net)

    return input_global,pointclouds_lab,pointclouds_higgs,pointclouds_bottom,pointclouds_top,pointclouds_W,pointclouds_Z,outputs
    """



inputs = BEST_PCT()
# print(len(inputs))
# for input in inputs: print(len(input))
# quit()
# print(inputs)
outputs = inputs[-1]
inputs = inputs[:-1]

# Callbacks
train_batches = provider.getMinEvents(TRAIN_FILE) // BATCH_SIZE # 3070017, 30700
test_batches  = provider.getMinEvents(TEST_FILE)  // BATCH_SIZE #  382583,  3825
#                       100,000   //  100   = 1000
BATCHES_PER_SUB_EPOCH = SUB_EPOCH_SIZE // BATCH_SIZE # 1000
sub_epochs_train = train_batches // BATCHES_PER_SUB_EPOCH # 30
sub_epochs_test  = test_batches  // BATCHES_PER_SUB_EPOCH # 3
# 30 train sub epochs, which is 30000 batches
# one sub epoch is 1000 batches, need 1000 eval batches
# 3825 batches in one full test file
# 8 full eval(8*3825=30600 batches) = more than one train epoch
# so each epoch, fill a list of 8 eval batches (make 1 like normal for all 6, do 8 times, append)
# make function that returns this list




class RecordMetrics(Callback):
    
    def on_epoch_end(self, epoch, logs=None):
        # Log keys: ['loss', 'accuracy', 'val_loss', 'val_accuracy', 'lr']
        # keys = list(logs.keys())
        # print("End epoch {} of training; got log keys: {}".format(epoch, keys))

        print('\n')
        log_string('************ EPOCH ' + str(epoch) + ' ************')    
        log_string("Learning Rate: " + str(logs["lr"]))
        log_string("Train Acc: " + str(logs["accuracy"]))
        log_string("Val Acc: " + str(logs ["val_accuracy"]) + '\n')
        log_string("Train Loss: " + str(logs["loss"]))
        log_string("Val Loss: " + str(logs["val_loss"]))
        log_string('Eval - Train = '  + str(logs["val_loss"]-logs["loss"]) )
        print('\n')

callbacks=[
    EarlyStopping(patience=EARLY_TOLERANCE, restore_best_weights=True),
    ReduceLROnPlateau(patience=5, verbose=1),
    ModelCheckpoint(os.path.join(LOG_DIR, 'epochs','{epoch}'), # save model every epoch
                    save_best_only=False, mode='auto', save_weights_only=False),
    ModelCheckpoint(os.path.join(LOG_DIR, 'bestModel'), verbose=1, # save best model
                    save_best_only=True, mode='auto', save_weights_only=False),
    RecordMetrics()
    # CustomCallback()
    # SubEpochSaver(getEvalIndices, provider.load_subEpoch )
]

        
model = Model(inputs=inputs,outputs=outputs)
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),              
              metrics=['accuracy'],
)

# print(model.summary())
# print(model.metrics_names)
train_batches = 1
test_batches = 1

hist =  model.fit(train_data,
                  epochs=MAX_EPOCH,
                  initial_epoch=0,
                  validation_data=test_data,
                  callbacks=callbacks,
                  verbose=1,
                #   steps_per_epoch = train_batches,
                #   validation_steps = test_batches,
                  steps_per_epoch = BATCHES_PER_SUB_EPOCH,
                  validation_steps = BATCHES_PER_SUB_EPOCH,
                #   use_multiprocessing = True,
)


# Accuracy and Loss plots
loss = [hist.history['loss'], hist.history['val_loss'] ]
acc  = [hist.history['accuracy'],  hist.history['val_accuracy']  ]
plotDir = os.path.join(LOG_DIR, 'plots')
os.makedirs(plotDir)
provider.plotPerformance(loss, acc, FLAGS.log_dir, plotDir)
dump(hist.history, os.path.join(plotDir,'history.joblib') )
print("Plotted PCT training Performance")

timeTaken = timedelta(seconds=int(time.time() - startTime))
log_string("\nFinished PCT, "+ FLAGS.log_dir + " took " + str( timeTaken ) + " to complete.\n")
LOG_FOUT.close()


"""
# This is from tensorflow documentation, helpful for desinging Custom Callbacks    
class CustomCallback(Callback):
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop testing; got log keys: {}".format(keys))

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop predicting; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))
"""


"""
def getEvalIndices(valInds = {}, sub_epoch_count = 0, is_epoch = False):

    if is_epoch: # do this once per full epoch
        inds = {i:[] for i in range(NUM_CLASSES)}
        for i in range(8): # 8 full eval events > one full train events
            for j, file in enumerate(TEST_FILE) :
                # inds[j] += list(provider.randomize_sub_epochs(np.arange(provider.getNevents(file))))
                inds[j] += list(provider.randomize_sub_epochs(np.arange(provider.getNevents(file))))
        return inds
    
    else:
        start = sub_epoch_count * SUB_EPOCH_SIZE
        end = start + SUB_EPOCH_SIZE        
        events = []
        for i in range(NUM_CLASSES):
            evalEvents = valInds[i][start:end]
            # evalEvents.sort()
            events.append(evalEvents)
        return events            

class SubEpochSaver(Callback):
    
    def __init__(self, getInds, subLoad):
        super(SubEpochSaver, self).__init__()
        self.getInds = getInds
        self.subLoad = subLoad
        self.sub_epoch_total_count = 0
        self.val_early_stop = np.inf
        self.train_early_stop = np.inf
        self.early_tol = 0

    def on_epoch_begin(self, epoch, logs=0):
        self.valInds = self.getInds(is_epoch = True)
        self.sub_epoch_time = time.time()
        self.sub_epoch_count = 0

    def on_train_batch_end(self, batch, logs=None):
        mid_cond = (batch > 0) and ((batch % BATCHES_PER_SUB_EPOCH) == 0)
        end_cond = batch == train_batches
        EVAL_SUB_EPOCH = mid_cond or end_cond
        if EVAL_SUB_EPOCH:

            print('\n')
            if end_cond: log_string("FINAL TRAIN BATCH")
            log_string('************ SUB EPOCH ' + str(self.sub_epoch_total_count) + ' ************')    

            events = self.getInds(valInds = self.valInds, sub_epoch_count = self.sub_epoch_count)

            subEpochDataset = self.subLoad(TEST_FILE, FRAMES, BATCH_SIZE, NUM_BES, events, mask)

            evalDict = self.model.evaluate(subEpochDataset,
                            return_dict = True, verbose = 0
                            )

            train_loss = logs["loss"]
            val_loss = evalDict["loss"]

            log_string("Train Batch: " + str(batch) )
            log_string("Val Acc: " + str(evalDict ["accuracy"]))
            log_string("Train Acc: " + str(logs["accuracy"]) + '\n')
            log_string("Val Loss: " + str(val_loss))
            log_string("Train Loss: " + str(train_loss))
            log_string('Eval - Train = '  + str(val_loss-train_loss) )


            # Save the model each epoch for debugging
            save_dir = os.path.join(LOG_DIR,'epochs', str(self.sub_epoch_total_count))
            # tf.keras.models.save_model(self.model, os.path.join(save_dir, 'model.ckpt') )
            # tf.keras.models.save_model(self.model, save_dir )
            self.model.save(save_dir+"model.h5")
            val_cond = val_loss < self.val_early_stop 
            train_cond = (train_loss < self.train_early_stop) and (train_loss > val_loss)
            model_improved = val_cond or train_cond
            if model_improved:
                if val_cond:   self.val_early_stop   = val_loss
                if train_cond: self.train_early_stop = train_loss
                self.earlytol = 0

                # Save the variables to disk.
                # tf.keras.models.save_model(self.model, os.path.join(LOG_DIR, 'model.ckpt') )
                # tf.keras.models.save_model(self.model, LOG_DIR)
                self.model.save(LOG_DIR+"model.h5")

                log_string("Val Condition: " + str(val_cond) + ", Train Conditon: " + str(train_cond))
                log_string("Model saved in file: %s" % LOG_DIR)
            else:            
                if self.earlytol >= EARLY_TOLERANCE:
                    log_string("Early stopping at Sub Epoch: " + self.sub_epoch_total_count)
                    self.model.stop_training = True
                else:
                    # print("No improvement for {0} epochs".format(earlytol))
                    log_string("No improvement for {0} Sub Epochs".format(self.earlytol))
                    self.earlytol+=1

            timeMessage = "\nSub Epoch " +str(self.sub_epoch_total_count)+ " Completed, took "+ str(timedelta(seconds = int(time.time() - self.sub_epoch_time))) + "\n"
            print(timeMessage)

            self.sub_epoch_time = time.time()
            self.sub_epoch_total_count += 1
            self.sub_epoch_count += 1
"""
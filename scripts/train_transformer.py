import argparse
from ctypes import BigEndianStructure
import math
import random
import subprocess
from datetime import datetime
from datetime import timedelta
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework import get_variables
import socket
import importlib
import os,ast
import sys
import time
from sklearn import metrics

startTime = time.time() # Tracks how long script takes


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR,'..', 'models')) 
sys.path.append(os.path.join(BASE_DIR,'..' ,'utils'))
import provider
import pct as MODEL

# plot everything on the log txt;
#   AUC for each cat vs. epochs; avg? (look at Brendans code?)
#   signal eff vs epochs
#   back eff vs. epochs
#   mean loss & val loss vs. epochs
#   


parser = argparse.ArgumentParser()


parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pct', help='Model name [default: pct]')
# parser.add_argument('--log_dir', default='log', help='Log dir [default: log in logs]')
parser.add_argument('--log_dir', default='1MLab', help='Log dir [default: log in logs]')
# parser.add_argument('--num_point', type=int, default=100, help='Point Number  [default: 100]')
parser.add_argument('--num_point', type=int, default=50, help='Point Number  [default: 50]')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 200]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate [default: 0.001]')
# parser.add_argument('--learning_rate', type=float, default=1e-6, help='Initial learning rate [default: 0.000001]')
parser.add_argument('--nevents', type=int, default=100000, help='Events to run [default: 100000]')


parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=5000000, help='Decay step for lr decay [default: 5000000]')
parser.add_argument('--wd', type=float, default=0.0, help='Weight Decay [Default: 0.0]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
# parser.add_argument('--data_dir', default='/pnfs/psi.ch/cms/trivcat/store/user/vmikuni/EMD_SF/', help='directory with data [default: hdf5_data]')
parser.add_argument('--data_dir', default='/uscms/home/bonillaj/nobackup/h5samples_ULv1/', help='directory with data [default: hdf5_data]')
parser.add_argument('--nfeat', type=int, default=13, help='Number of features PF [default: 13]')
parser.add_argument('--ncat', type=int, default=6, help='Number of categories [default: 6]')
# parser.add_argument('--ncat', type=int, default=2, help='Number of categories [default: 2]')
# parser.add_argument('--sample', default='qg', help='sample to use')
parser.add_argument('--sample', default='best', help='sample to use')
parser.add_argument('--simple', action='store_true', default=False,help='Use simplified model')

FLAGS = parser.parse_args()
DATA_DIR = FLAGS.data_dir
SAMPLE = FLAGS.sample
EPOCH_CNT = 0
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
NUM_FEAT = FLAGS.nfeat
NUM_CLASSES = FLAGS.ncat
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
# BIGBATCH_SIZE = 10016
BIGBATCH_SIZE = BATCH_SIZE * 500
NUM_EVENTS = FLAGS.nevents
# NUM_EVENTS = -1
# NUM_EVENTS = 500000
# NUM_EVENTS = 1000000
# NUM_EVENTS = 1000
if (NUM_EVENTS > 0):
    if (NUM_EVENTS < BIGBATCH_SIZE): BIGBATCH_SIZE = NUM_EVENTS # Make sure a batch is not larger than nevents 

    batchRemain = NUM_EVENTS % BATCH_SIZE
    if batchRemain != 0:  NUM_EVENTS += (BATCH_SIZE - batchRemain) # Make sure nevts is divisible by batch size

    bigbatchRemain = NUM_EVENTS % BIGBATCH_SIZE
    if bigbatchRemain != 0: NUM_EVENTS += (BIGBATCH_SIZE - bigbatchRemain) # Add events to get even batches


MODEL_FILE = os.path.join(BASE_DIR, '..', 'models',FLAGS.model+'.py')
LOG_DIR = os.path.join('../logs/',FLAGS.log_dir)
print("\nLOG DIR: " + LOG_DIR + "\n")

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
if not os.path.exists(LOG_DIR+"/epochs/"): os.makedirs(LOG_DIR+"/epochs/")
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train_transformer.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

LEARNING_RATE_CLIP = 1e-6
HOSTNAME = socket.gethostname()
EARLY_TOLERANCE=15
multi = False

# scale = "newBEST_Basic"

if SAMPLE == 'best':
    multi = True
    # TRAIN_FILE = [os.path.join(DATA_DIR, mySamp+'Sample_2017_BESTinputs_train_flattened_standardized.h5') for mySamp in ["WW","ZZ","HH","TT","QCD","BB"]]
    # TEST_FILE  = [os.path.join(DATA_DIR, mySamp+'Sample_2017_BESTinputs_validation_flattened_standardized.h5') for mySamp in ["WW","ZZ","HH","TT","QCD","BB"]]
    # TRAIN_FILE = [os.path.join(DATA_DIR, mySamp+'Sample_2017_BESTinputs_train_flattened_'+scale+'.h5') for mySamp in ["WW","ZZ","HH","TT","BB","QCD"]]
    # TEST_FILE  = [os.path.join(DATA_DIR, mySamp+'Sample_2017_BESTinputs_validation_flattened_'+scale+'.h5') for mySamp in ["WW","ZZ","HH","TT","BB","QCD"]]
    TRAIN_FILE = [os.path.join(DATA_DIR, mySamp+'Sample_2017_BESTinputs_train_flattened.h5') for mySamp in ["WW","ZZ","HH","TT","BB","QCD"]]
    TEST_FILE  = [os.path.join(DATA_DIR, mySamp+'Sample_2017_BESTinputs_validation_flattened.h5') for mySamp in ["WW","ZZ","HH","TT","BB","QCD"]]
else:
    sys.exit("ERROR: SAMPLE NOT FOUND")
    
def progress_bar(batchesRan, batchesTotal, batchBegin, timeAvg):
    # Event Progress:
    eventsRan   = batchesRan*BATCH_SIZE
    eventsTotal = batchesTotal*BATCH_SIZE
    eventString = str(eventsRan) + " / " + str(eventsTotal)
    
    batchRatio  = float(batchesRan) / float(batchesTotal) 
    ratioString = "(" + str(batchRatio * 100.)[:4] + "%) "

    # Create the progress bar:
    progN   = int(batchRatio * 30)
    arrowN  = (progN>1) and (progN<30)
    equalN  = progN - arrowN
    periodN = 30 - progN 
    progString = " [" + "="*equalN + ">"*arrowN + "."*periodN + "] "

    # Calculate ETA:
    batchTime  = time.time() - batchBegin
    new_timeAvg = timeAvg + ( (batchTime - timeAvg)/float(batchesRan) ) # Add value to average
    etaSeconds = new_timeAvg * float(batchesTotal - batchesRan)
    etaString  = "ETA: " + str(timedelta(seconds = int(etaSeconds))) # int gets rid of microseconds
    

    print("\r" + eventString + progString + ratioString + etaString, end='\t')
    return new_timeAvg



def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, LEARNING_RATE_CLIP) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    g = tf.Graph()
    run_meta = tf.RunMetadata()
    with g.as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, mask_pl,labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT,NUM_FEAT)

            is_training = tf.placeholder(tf.bool, shape=())

            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)
            print("--- Get model and loss")

            print(tf.__version__)

            if FLAGS.simple:
                pred,att1,att2,att3 = MODEL.get_model_simple(pointclouds_pl,mask_pl, 
                                                             is_training=is_training,
                                                             scname='PL',
                                                             num_class=NUM_CLASSES,
                                                             bn_decay=bn_decay, weight_decay=FLAGS.wd)
            else:
                pred,att1,att2,att3 = MODEL.get_model(pointclouds_pl,mask_pl, 
                                                      is_training=is_training,
                                                      scname='PL',
                                                      num_class=NUM_CLASSES,
                                                      bn_decay=bn_decay, weight_decay=FLAGS.wd)

            

            loss = MODEL.get_loss(pred, labels_pl,NUM_CLASSES)            
            pred = tf.nn.softmax(pred)

            tf.summary.scalar('loss', loss)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':                
                optimizer = tf.train.AdamOptimizer(learning_rate)

            train_op = optimizer.minimize(loss, global_step=batch)

            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep = 0)
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        
        
        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        
        log_string("Total number of weights for the model: " + str(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        
        ops = {
            'pointclouds_pl': pointclouds_pl,
            'labels_pl': labels_pl,
            'mask_pl':mask_pl,
            'attention':att1,

            
            'is_training': is_training,
            'pred': pred,


            'loss': loss,
            'train_op': train_op,
            'learning_rate':learning_rate,
            'merged': merged,
            'step': batch,
        }

        early_stop = np.inf
        earlytol = 0

        for epoch in range(MAX_EPOCH):
            preEpochTime = time.time()
            
            log_string('**** EPOCH %03d ****' % (epoch))
            print(FLAGS.log_dir)
            sys.stdout.flush()            

            train_one_epoch(sess, ops, train_writer) 
            
            trainEpochTime = time.time()
            timeTaken = (trainEpochTime - preEpochTime) / 3600.
            timeMessage = "Trained Epoch, took "+ str(timeTaken)[:5] + " hours to complete.\n"
            print(timeMessage)

            lss = eval_one_epoch(sess, ops, test_writer)

            evalEpochTime = time.time()
            timeTaken = (evalEpochTime - trainEpochTime) / 3600.
            timeMessage = "Evaluated Epoch, took "+ str(timeTaken)[:5] + " hours to complete.\n"
            print(timeMessage)

            # Save the model each epoch for debugging
            epoch_path = os.path.join(LOG_DIR,'epochs',str(epoch))
            save_path_all = saver.save(sess, os.path.join(epoch_path, 'model.ckpt'), global_step=epoch)

            cond = lss < early_stop 
            if cond:
                early_stop = lss
                earlytol = 0
                # Save the variables to disk.

                save_path = saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'))
                log_string("Model saved in file: %s" % LOG_DIR)
            else:            
                if earlytol >= EARLY_TOLERANCE:
                    break
                else:
                    # print("No improvement for {0} epochs".format(earlytol))
                    log_string("No improvement for {0} epochs".format(earlytol))
                    earlytol+=1
            

def get_batch(data_pl,label, start_idx, end_idx):
    batch_label = label[start_idx:end_idx]
    batch_data_pl = data_pl[start_idx:end_idx,:,:]
    return batch_data_pl, batch_label


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training=True
    
    loss_sum = 0
    #this loads all the data, then works on it in batches.
    ###############Implement a randomizer for epoch for non all stats. dont pull the same events each time.
    #               load Nevents total, divide by NUM_EVENTS to run, get random int from ratio, mult to startbigidx
    # (10001 // 1000 = 10, get random int from 0-10, then mult with start_bigidx, get endbigidx by adding BIGBatch to startbigidx or just using startbigidx  )
    # SIMPLER IMPLEMENTATION: first line of for loop: bigBatchidx + randint (note: if ratio = N, randint= 0->N-1)
    # check with johan if the events are ordered by pT. if so, this will need to be more complicated.  

    file_bigSize = provider.getNevents(TRAIN_FILE[0])
    max_BigBatches = int(file_bigSize // BIGBATCH_SIZE)
    if NUM_EVENTS == -1: num_bigBatches = max_BigBatches  
    else:                num_bigBatches = int(NUM_EVENTS // BIGBATCH_SIZE)
    
    # batchDiff will shift the batch index, calling differ, but the same amount of, events from the h5 files
    # This is 0 if num_BigBatches = max_BigBatches, no effect in this case (allStats)
    batchDiff = max_BigBatches - num_bigBatches
    # Produces an integer 0 <= batchShift <= batchDiff
    batchShift = random.randint(0,batchDiff)
    # print(file_bigSize, BIGBATCH_SIZE, max_BigBatches, num_bigBatches)
    # print(batchDiff, batchShift, range(batchShift, num_bigBatches + batchShift))

    # file_size = current_data_pl.shape[0]
    num_batches = (BIGBATCH_SIZE * NUM_CLASSES) // BATCH_SIZE
    #num_batches = 4
    batchesTotal = num_bigBatches * num_batches
    # eventsTotal  = batchesTotal * BATCH_SIZE

    # if NUM_EVENTS == -1: file_bigSize = provider.getNevents(TRAIN_FILE[0]) 
    # else:           file_bigSize = NUM_EVENTS 
    # num_bigBatches = int(file_bigSize // BIGBATCH_SIZE)

    # # file_size = current_data_pl.shape[0]
    # num_batches = (BIGBATCH_SIZE * NUM_CLASSES) // BATCH_SIZE
    # #num_batches = 4
    # eventsTotal = num_bigBatches * num_batches * BATCH_SIZE

    log_string(str(datetime.now()))
    batchesRan = 0
    timeAvg = 0
    # for bigBatch_idx in range(num_bigBatches):
    for bigBatch_idx in range(batchShift, num_bigBatches + batchShift):
        start_bigidx = bigBatch_idx * (BIGBATCH_SIZE)
        end_bigidx = (bigBatch_idx+1) * (BIGBATCH_SIZE)
    
        current_data_pl, current_label = provider.load_h5BEST(TRAIN_FILE, NUM_POINT, start_bigidx, end_bigidx)

        current_data_pl, current_label, _ = provider.shuffle_data(current_data_pl, np.squeeze(current_label))
        
        for batch_idx in range(num_batches):
            batchBegin = time.time()

            start_idx = batch_idx * (BATCH_SIZE)
            end_idx = (batch_idx+1) * (BATCH_SIZE)
            batch_data_pl, batch_label = get_batch(current_data_pl, current_label,start_idx, end_idx)
            mask_padded = batch_data_pl[:,:,2]==0


            
            feed_dict = {             
                ops['pointclouds_pl']: batch_data_pl,
                ops['labels_pl']: batch_label,
                ops['mask_pl']: mask_padded.astype(float),
                ops['is_training']: is_training,
            }
            
            train_op = 'train_op'
            attention = 'attention'
            loss = 'loss'
            # print(loss)
            summary, step, _, loss,attention = sess.run([ops['merged'], ops['step'],
                                                        ops['train_op'],
                                                        ops['loss'],ops['attention']
                                                    ],
                                                        feed_dict=feed_dict)
            # print(loss, np.mean(loss))
            #print(attention)
            train_writer.add_summary(summary, step)
            loss_sum += np.mean(loss)

            # Keep track of progress
            if batchesRan == 0: print("Training Events...")
            batchesRan += 1
            timeAvg = progress_bar(batchesRan, batchesTotal, batchBegin, timeAvg)
    print("")
    print("BatchesTotal, BatchesRan: ", batchesTotal, batchesRan)
    print(loss_sum, batchesRan)
    print('\n')
    log_string('mean loss: %f' % ((loss_sum*1.0) / float(batchesRan)))

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    loss_sum = 0
    y_source=[]
    every_label=[]

    file_bigSize = provider.getNevents(TEST_FILE[0])
    max_BigBatches = int(file_bigSize // BIGBATCH_SIZE)
    if NUM_EVENTS == -1: num_bigBatches = max_BigBatches  
    else:                num_bigBatches = int(NUM_EVENTS // BIGBATCH_SIZE)
    # if (NUM_EVENTS < file_bigSize) and (NUM_EVENTS > 0): file_bigSize = NUM_EVENTS 
    # num_bigBatches = int(file_bigSize // BIGBATCH_SIZE)


    # batchDiff will shift the batch index, calling differ, but the same amount of, events from the h5 files
    # This is 0 if num_BigBatches = max_BigBatches, no effect in this case (allStats)
    batchDiff = max_BigBatches - num_bigBatches
    # Produces an integer 0 <= batchShift <= batchDiff
    batchShift = random.randint(0,batchDiff)
    # print(file_bigSize, BIGBATCH_SIZE, max_BigBatches, num_bigBatches)
    # print(batchDiff, batchShift, range(batchShift, num_bigBatches + batchShift))

    # file_size = current_data_pl.shape[0]
    num_batches = (BIGBATCH_SIZE * NUM_CLASSES) // (BATCH_SIZE)
    #num_batches = 4
    batchesTotal = num_bigBatches * num_batches
    # eventsTotal  = batchesTotal * BATCH_SIZE
        
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))

    batchesRan = 0
    timeAvg = 0
    # lossList = [[] for i in range(6)]
    # for bigBatch_idx in range(num_bigBatches):
    for bigBatch_idx in range(batchShift, num_bigBatches + batchShift):

        # print(batchesRan)
        start_bigidx = bigBatch_idx * (BIGBATCH_SIZE)
        end_bigidx = (bigBatch_idx+1) * (BIGBATCH_SIZE)
    
        current_data_pl, current_label = provider.load_h5BEST(TEST_FILE, NUM_POINT, start_bigidx, end_bigidx)
        # current_data_pl, current_label = provider.load_h5BEST(TRAIN_FILE, NUM_POINT, start_bigidx, end_bigidx)
        # current_data_pl, current_label = provider.load_h5BEST(TEST_FILE, NUM_POINT, start_bigidx, end_bigidx, debug=True)

        current_data_pl, current_label, _ = provider.shuffle_data(current_data_pl, np.squeeze(current_label))

        # for i, thisData in enumerate(current_data_pl):
        #     thisLabel = current_label[i]
        #     thisLoss  = lossList[i]

        for batch_idx in range(num_batches):
            batchBegin = time.time()

            start_idx = batch_idx * (BATCH_SIZE)
            end_idx = (batch_idx+1) * (BATCH_SIZE)
            # batch_data_pl, batch_label = get_batch(thisData, thisLabel,start_idx, end_idx)
            # if len(batch_data_pl) < 32: continue
            batch_data_pl, batch_label = get_batch(current_data_pl, current_label,start_idx, end_idx)
            mask_padded = batch_data_pl[:,:,2]==0
            feed_dict = {             
                ops['pointclouds_pl']: batch_data_pl,
                ops['labels_pl']: batch_label,
                ops['is_training']: is_training,
                ops['mask_pl']: mask_padded.astype(float),
            }
                
            if (batch_idx == 0) and (bigBatch_idx == 0): start_time = time.time()

            loss = 'loss'
            # print(loss)
            summary, step, loss,pred,lr = sess.run([ops['merged'], ops['step'],
                                                    ops['loss'],   ops['pred'],
                                                    ops['learning_rate']
                                                    ],
                                                feed_dict=feed_dict)
            # print(loss, np.mean(loss))            

            if (batch_idx == 0) and (bigBatch_idx == 0):
                duration = time.time() - start_time
                log_string("Single Batch Eval time: "+str(duration)) 
                log_string("Single Batch Learning rate: "+str(lr)) 
                #log_string("{}".format(sub_feat))


            test_writer.add_summary(summary, step)
            # thisLoss.append(loss)
            loss_sum += np.mean(loss) 
            if len(y_source)==0: 
                y_source = np.squeeze(pred)
                every_label = np.array(batch_label)
            else:                
                y_source = np.concatenate((y_source,np.squeeze(pred)),axis=0)
                every_label = np.concatenate((every_label,batch_label), axis=0)
            
            # Keep track of progress
            if batchesRan == 0: print("Evaluating Events...")
            batchesRan += 1
            timeAvg = progress_bar(batchesRan, batchesTotal, batchBegin, timeAvg)
    print("")
    print("BatchesTotal, BatchesRan: ", batchesTotal, batchesRan)
    print(loss_sum, batchesRan)
    print('\n')            


    name_convert = { 0:'W', 1:'Z', 2:'H', 3:'T', 4:'B', 5:'QCD' }
    
    # log_string("Loss List:")
    # for i, loss in enumerate(lossList): log_string(str(name_convert[i]) +': '+ str(loss))
    # log_string("Loss Means:")
    # for i, loss in enumerate(lossList): log_string(str(name_convert[i]) +': '+ str(np.mean(loss)))

    #fix evaluate code
    #add SV
    #add frames
    # label = every_label[:batchesRan*(BATCH_SIZE)]


    for isample in np.unique(every_label):
        fpr, tpr, _ = metrics.roc_curve(every_label==isample, y_source[:,isample], pos_label=1)    
        log_string("Class: {}, AUC: {}".format(name_convert[isample],metrics.auc(fpr, tpr)))
        bineff = np.argmax(fpr>0.1)
        log_string('SOURCE: effS at {0} effB = {1}'.format(tpr[bineff],fpr[bineff]))
    log_string('mean loss: %f' % ((loss_sum*1.0) / float(batchesRan)))

    EPOCH_CNT += 1


    return ((loss_sum*1.0) / float(batchesRan))
    


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()

    timeTaken = (time.time() - startTime) / 86400.
    log_string("\nFinished PCT, "+ FLAGS.log_dir + " total time was " + str( timeTaken )[:5] + "days to complete.\n")
    LOG_FOUT.close()

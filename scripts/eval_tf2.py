import argparse
from asyncio import events
import h5py
from math import *
# from scripts.train_transformer import SUB_EPOCH_NUM, SUB_EPOCH_SIZE
import tensorflow as tf
import numpy as np
import os
import sys
import time
from datetime import timedelta
import imp

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout

#np.set_printoptions(threshold=sys.maxsize)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR,'..', 'models'))
sys.path.append(os.path.join(BASE_DIR,'..' ,'utils'))
import provider
import pct_tf2 as MODEL

# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPUs to use [default: 0]')
parser.add_argument('--model_path', default='', help='Model checkpoint path')
parser.add_argument('--batch', type=int, default=32, help='Batch Size  during training [default: 32]')
# parser.add_argument('--num_point', type=int, default=50, help='Point Number [default: 50]')
parser.add_argument('--data_dir', default='/uscms_data/d3/bonillaj/h5samples_ULv1/', help='directory with data [default: hdf5_data]')
# parser.add_argument('--nfeat', type=int, default=13, help='Number of features [default: 13]')
parser.add_argument('--ncat', type=int, default=6, help='Number of categories [default: 6]')
parser.add_argument('--name', default="", help='name of the output file')
parser.add_argument('--h5_folder', default="../h5/", help='folder to store output files')
# parser.add_argument('--sample', default='best', help='sample to use')
parser.add_argument('--simple', action='store_true', default=False,help='Use simplified model')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log in logs]')
parser.add_argument('--epoch', default='', help='Epoch to Evaluate [default: '']')
# parser.add_argument('--frames', default=['PF_cands_LabFrame'], help='Frames to run [default: [PF_cands_LabFrame] ]')

FLAGS = parser.parse_args()
MODEL_PATH = FLAGS.model_path
DATA_DIR = FLAGS.data_dir
H5_DIR = os.path.join(BASE_DIR, DATA_DIR)
H5_OUT = FLAGS.h5_folder
EVAL_EPOCH = FLAGS.epoch 
NAME  = FLAGS.name
NUM_CLASSES = FLAGS.ncat
# NUM_FEAT = FLAGS.nfeat
NUM_FEAT = 13, 9
# NUM_POINT = FLAGS.num_point
NUM_POINT = 50, 10

# MAIN SCRIPT
BATCH_SIZE = FLAGS.batch


FRAMES, DEP_FRAMES = provider.getFrames(FLAGS.log_dir)
print(FRAMES)

simple = False
if "S" in FLAGS.log_dir: simple = True

if not os.path.exists(H5_OUT): os.mkdir(H5_OUT)  
LOG_DIR = os.path.join('../logs/',FLAGS.log_dir)
# LOG_DIR = os.path.join('../logs/',MODEL_PATH,FLAGS.log_dir)
if not os.path.exists(LOG_DIR): 
    print('LOG_DIR does not exist:',LOG_DIR)
    quit()

# Load the pct model copied during training 
# MODEL = imp.load_source('pct', os.path.join(LOG_DIR, 'pct.py'))    
# ep = "26"
# ep = "25"
# ep = "15"
# LOG_DIR = os.path.join(LOG_DIR,"epochs",ep)
if NAME == "": NAME = FLAGS.log_dir
if not EVAL_EPOCH == "": 
    NAME += "_E" + EVAL_EPOCH
    LOG_DIR = os.path.join(LOG_DIR,"epochs",EVAL_EPOCH)
# LOG_DIR = os.path.join(LOG_DIR,"epochs",ep,'variables')

SAMPLES = ["WW","ZZ","HH","TT","BB","QCD"]
MASK=[]
NUM_BES = None
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

print('#### Batch Size : {0}'.format(BATCH_SIZE))
print('#### Log Dir: ' + LOG_DIR)
print('#### H5 File Name: ' + NAME)

print('### Starting evaluation')
EVALUATE_FILE = [os.path.join(DATA_DIR, mySamp+'Sample_2017_BESTinputs_test_flattened.h5') for mySamp in SAMPLES]

N_EVENTS = [provider.getNevents(file) for file in EVALUATE_FILE]
total_batches = np.sum([ev // BATCH_SIZE for ev in N_EVENTS])
# batches = np.sum(N_EVENTS) // BATCH_SIZE
# idx = [np.arange(ev) for ev in N_EVENTS]

def load_bes_eval():  

    # tf will call generator again when all events are exhausted
    # needs to work for train and eval
    # last batch should grab remaining events for all files
    # this gen is passed to to dataset.from_generator
    def gen_eval():
        for i, events in enumerate(N_EVENTS):
            batches = events // BATCH_SIZE
            idx = np.arange(events)
            # file = EVALUATE_FILE[i]

            for batch in range(batches):
                start = batch * BATCH_SIZE
                end = (batch+1) * BATCH_SIZE
                if batch + 1 == batches: end = None
                
                data = load_h5BEST_eval(i, idx[start:end])

                yield tuple([data[frame] for frame in FRAMES])

    if ("BES_vars" in FRAMES):
        dataset = tf.data.Dataset.from_generator(
                gen_eval,
                output_signature=(
                    tuple( [tf.TensorSpec(shape=(None, 50, 13), dtype=tf.float64) for i in range(len(FRAMES) - 1)]
                            + [tf.TensorSpec(shape= (None, NUM_BES), dtype=tf.float64)] ) )
                )
    else:
        dataset = tf.data.Dataset.from_generator(
                gen_eval, 
                output_signature=(
                    tuple( [tf.TensorSpec(shape=(None, 50, 13), dtype=tf.float64) for i in range(len(FRAMES))] ) )
                )
    
    return dataset

def load_h5BEST_eval(i, events):
    """ Load in BEST data from h5 Files.
        Input:
        list of h5 filenames, desired frames, number of PF cands, list of lists for random events to load
        Return:
        data dictionary, labels
        data dictionary structure; for each frame, an array of: 
            Dep Frames: BATCH_SIZE x PF Cands(num_points) x vars (13)
            SV_vars:    BATCH_SIZE x Sec. Verts. (10)     x SV vars (9)
            BES_vars:   BATCH_SIZE x BES_vars (depends on input frames)
                Note: BATCH_SIZE = SUB_EPOCH_SIZE * NUM_CLASSES, typically = 100 * 6 = 600
             
    """
    #==================================================================================
    # Initialize File List, Data Structures, Load Invariant Frame, Create Labels   ////
    #==================================================================================
    file = EVALUATE_FILE[i]
    samp = SAMPLES[i]
    # Grab h5 File
    f = h5py.File(file, 'r')

    # Initialize data dictionary
    data = dict()
    events = list(events)
    # Load the Invariant Frame Data:
    # Frame invariant shape: N_events x N_PFCands(50) x N_vars(9)
    # h5 Var indices: 0-PUPPI_Weights, 1-abs(pdgID), 2-charge, 3-isChargedHadron, 4-isElectron, 5-isMuon, 6-isNeutralHadron, 7-isPhoton, 8-pdgID
    data["PF_cands_AllFrame"] = f["PF_cands_AllFrame"][events][..., [2,4,5,3,6,7] ] 
    # ^This data structure is BATCH_SIZE x N_PFCands x [charge, isElectron, isMuon, isChargedHadron, isNeutralHadron, isPhoton](6)

    #==================================================================================
    # Load h5 BEST Data ///////////////////////////////////////////////////////////////
    #==================================================================================
    """
    # Four types of h5 data structures; Frame Invariant, Frame Dependent, Secondary Vertex, and BES_Vars
    
    # BES_Vars data is used after the normal PCT training;
    # Simply load the mask containing the desired frames, then load the appropriate BES_vars

    # First two PCT input variables NEED to be deltaEta, then deltaPhi (for the knn)
    # Third PCT input variable is used for 'mask_padded'. Choose something that is NEVER 0 when it is filled. (deltaR, SV_nTracks)
    
    # Frame Invariant vars (isMuon, charge, etc.) will be added to each Frame Dependent PCT input array in the train_transformer code (to load INV data once)
    # 13 vars total:
    # Frame Dependent: [deltaEta, deltaPhi, deltaR, logEnergyRatio, logpTRatio, logpT, logEnergy](7), 
    #   + Frame Invariant: [charge, isElectron, isMuon, isChargedHadron, isNeutralHadron, isPhoton](6)

    # Secondary Vertex vars will be a seperate "frame"; Will need to 'scale' energy/mass and calculate deltas & logRatios;
    # 9 vars total:
    # [deltaEta, deltaPhi, nTracks, Ndof, chi2, logMassRatio, logpTRatio, logMass, logpT ]
    """

    for frame in FRAMES:
        # frame = str(myFrame, 'utf-8')
        # print(frame)
        if frame == "SV_vars":
            # Secondary Vertex shape: N_events x N_SV(10) x N_vars(7)
            # h5 Var indices: 0-SV_Ndof, 1-SV_chi2, 2-SV_eta, 3-SV_mass, 4-SV_nTracks, 5-SV_phi, 6-SV_pt

            data_sv = [f[frame][events[i],:,:] for i,f in enumerate(fs)]
            
            data_sv_etaPhi = [ arr[:,:][...,[2,5]] for arr in data_sv] # Use this to calc delta eta/ delta phi

            with np.errstate(divide = 'ignore'): # The unfilled entries will raise a warning (log(0) = -inf). We mask them later, so ignore the warning.
                data_sv_logPtMass = [ np.log(arr[:,:][...,[3,6]]) for arr in data_sv] # Use this to calc log energy/mass ratios 

            data_sv_remain = [ arr[:,:][...,[4,0,1]] for arr in data_sv] # Remaining vars; Use nTracks for padded mask, so it is first.

            del data_sv # delete this now that we have extracted what we need

            # Need to create deltaEta, deltaPhi, logMassRatio, logpTRatio

            # Load jetAK8 data; indices: 545-jetAK8_eta, 546-jetAK8_mass, 547-jetAK8_phi, 548-jetAK8_pt
            data_jet = [ f["BES_vars"][events[i]][...,[545,546,547,548]] for i,f in enumerate(fs) ]

            data_sv_logRatios = []
            # Use jetAK8 data to calc deltas and logRatios, then delete it
            for i, arr in enumerate(data_jet): 
                data_sv_etaPhi[i] -= arr[:][...,np.newaxis,[0,2]] 
                data_sv_logRatios.append(data_sv_logPtMass[i] - np.log(arr[:][...,np.newaxis,[1,3]]))
            del data_jet

            # Put finished data together, concatenate across h5 files, delete obsolete arrays
            data[frame] = np.concatenate( [ np.concatenate( (data_sv_etaPhi[i], data_sv_remain[i], 
                                data_sv_logRatios[i], data_sv_logPtMass[i]), axis=2 ) for i in range(len(fs)) ] )

            del data_sv_etaPhi; del data_sv_remain; del data_sv_logRatios; del data_sv_logPtMass
            # print("data", np.array(data_sv).shape, np.array(data_sv[0]).shape, np.array(data_sv[0][0]).shape, np.array(data_sv[0][0][0]).shape)

        elif frame == "BES_vars":
            h5File = "/uscms/home/bonillaj/nobackup/h5samples_ULv1/" + samp + "Sample_2017_BESTinputs_test_flattened_newBEST_Basic.h5" 
            data[frame] = h5py.File(h5File,'r')[frame][events][:,MASK] 
                                            

        elif frame in DEP_FRAMES:
            # Frame dependent shape: N_events x N_PFCands(50) x N_vars(11)
            # h5 Var indices: 0-deltaEta, 1-deltaPhi, 2-deltaR, 3-energy, 4-logEnergy, 5-logEnergyRatio, 6-logpT, 7-logpTRatio, 8-px, 9-py, 10-pz
            
            # This data structure is BATCH_SIZE x N_PFCands x [deltaEta, deltaPhi, deltaR, logEnergyRatio, logpTRatio, logpT, logEnergy],
            #  concatenated with the Invariant variables, then concatenated across the h5 samples
            data[frame] = np.concatenate( (f[frame][events][..., [0,1,2,5,7,6,4] ], data["PF_cands_AllFrame"]), axis=2 )  

        else:
            print("FRAME ERROR: Invalid frame: " + frame)
            quit()

    del data["PF_cands_AllFrame"] # delete this now that we are done

    return data

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

# # Create Truth arrays; shape: N_events x 6
# truthArrays = [np.zeros( (ev, len(N_EVENTS)) ) for ev in N_EVENTS] 

# # Arrays are filled with zeros. Now set 1's to record Truth particle info
# for i in range(len(SAMPLES)):
#     truthArrays[i][:,i] = 1.

# labels = np.concatenate(truthArrays)



inputs = BEST_PCT()

outputs = inputs[-1]
inputs = inputs[:-1]

model = Model(inputs=inputs,outputs=outputs)
checkpoint = tf.train.Checkpoint(model)
checkpoint.restore(LOG_DIR)
eval_data  = load_bes_eval()

# Predict
# pctModel = load_model(LOG_DIR)
BESpredict = model.predict(eval_data, batch_size=BATCH_SIZE, verbose=1, steps=total_batches )
print("Made predictions using the neural network")

labels = np.concatenate([np.full(nevents,i) for i, nevents in enumerate(N_EVENTS)] )
with h5py.File(os.path.join(H5_OUT, NAME+'.h5'), "w") as fh5:
    dset = fh5.create_dataset("DNN", data=BESpredict)
    dset = fh5.create_dataset("pid", data=labels)
print(BESpredict.shape, labels.shape)

################################################          

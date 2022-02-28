import argparse
import h5py
from math import *
import tensorflow as tf
import numpy as np
import os, ast
import sys
from sklearn import metrics

#np.set_printoptions(threshold=sys.maxsize)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR,'..', 'models'))
sys.path.append(os.path.join(BASE_DIR,'..' ,'utils'))
#from MVA_cfg import *
import provider
import pct as MODEL

# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPUs to use [default: 0]')
parser.add_argument('--model_path', default='', help='Model checkpoint path')
parser.add_argument('--batch', type=int, default=64, help='Batch Size  during training [default: 64]')
parser.add_argument('--num_point', type=int, default=50, help='Point Number [default: 50]')
parser.add_argument('--data_dir', default='/uscms/home/bonillaj/nobackup/h5samples_ULv1/', help='directory with data [default: hdf5_data]')
parser.add_argument('--nfeat', type=int, default=13, help='Number of features [default: 13]')
parser.add_argument('--ncat', type=int, default=6, help='Number of categories [default: 6]')
parser.add_argument('--name', default="", help='name of the output file')
parser.add_argument('--h5_folder', default="../h5/", help='folder to store output files')
parser.add_argument('--sample', default='best', help='sample to use')
parser.add_argument('--simple', action='store_true', default=False,help='Use simplified model')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log in logs]')

FLAGS = parser.parse_args()
MODEL_PATH = FLAGS.model_path
DATA_DIR = FLAGS.data_dir
H5_DIR = os.path.join(BASE_DIR, DATA_DIR)
H5_OUT = FLAGS.h5_folder
if not os.path.exists(H5_OUT): os.mkdir(H5_OUT)  
# LOG_DIR = os.path.join('../logs/',FLAGS.log_dir)
LOG_DIR = os.path.join('../logs/',MODEL_PATH,FLAGS.log_dir)
if not os.path.exists(LOG_DIR): 
    print('LOG_DIR does not exist:',LOG_DIR)
    quit()

# MAIN SCRIPT
NUM_POINT = FLAGS.num_point
BATCH_SIZE = FLAGS.batch
NFEATURES = FLAGS.nfeat
SAMPLE = FLAGS.sample

NUM_CATEGORIES = FLAGS.ncat
#Only used to get how many parts per category

print('#### Batch Size : {0}'.format(BATCH_SIZE))
print('#### Point Number: {0}'.format(NUM_POINT))
print('#### Using GPUs: {0}'.format(FLAGS.gpu))


    
print('### Starting evaluation')

multi = True
# EVALUATE_FILE = os.path.join(DATA_DIR, 'eval_multi_100P_Jedi.h5')
# EVALUATE_FILE = [os.path.join(DATA_DIR, mySamp+'Sample_2017_BESTinputs_test_flattened_standardized.h5') for mySamp in ["WW","ZZ","HH","TT","QCD","BB"]]
# EVALUATE_FILE = [os.path.join(DATA_DIR, mySamp+'Sample_2017_BESTinputs_test_flattened_'+scale+'.h5') for mySamp in ["WW","ZZ","HH","TT","BB","QCD"]]
EVALUATE_FILE = [os.path.join(DATA_DIR, mySamp+'Sample_2017_BESTinputs_test_flattened.h5') for mySamp in ["WW","ZZ","HH","TT","BB","QCD"]]



  
def eval():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(FLAGS.gpu)):
            pointclouds_pl, mask_pl,labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT,NFEATURES)
          
            batch = tf.Variable(0, trainable=False)
                        
            is_training = tf.placeholder(tf.bool, shape=())
            if FLAGS.simple:                
                pred,atts1,atts2,atts3 = MODEL.get_model_simple(pointclouds_pl,mask_pl, is_training=is_training,scname='PL',num_class=NUM_CATEGORIES)           
            else:
                pred,atts1,atts2,atts3 = MODEL.get_model(pointclouds_pl,mask_pl, is_training=is_training,scname='PL',num_class=NUM_CATEGORIES)           
            pred=tf.nn.softmax(pred)            
            saver = tf.train.Saver()
          

    
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)

        # saver.restore(sess,os.path.join(LOG_DIR,MODEL_PATH,'model.ckpt'))
        saver.restore(sess,os.path.join(LOG_DIR,'model.ckpt'))
        print('model restored')
        
        

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'mask_pl': mask_pl,
               'is_training': is_training,
               'pred': pred,
               'atts1': atts1,
               'atts2': atts2,
               'atts3': atts3,
        }
            
        eval_one_epoch(sess,ops)

def get_batch(data_pl,label, start_idx, end_idx):
    batch_label = label[start_idx:end_idx]
    batch_data_pl = data_pl[start_idx:end_idx,:,:]
    return batch_data_pl, batch_label
        
def eval_one_epoch(sess,ops):
    is_training = False
    y_pred = []
    print("Evaluating...")
    current_data_pl, current_label = provider.load_h5BEST(EVALUATE_FILE, NUM_POINT, 0, 10000)

    file_size = current_data_pl.shape[0]
    num_batches = file_size // BATCH_SIZE        
    #num_batches = 4
    for batch_idx in range(num_batches):
        start_idx = batch_idx * (BATCH_SIZE)
        end_idx = (batch_idx+1) * (BATCH_SIZE)
        batch_data_pl, batch_label = get_batch(current_data_pl, current_label,start_idx, end_idx)
        mask_padded = batch_data_pl[:,:,2]==0
        
        feed_dict = {             
            ops['pointclouds_pl']: batch_data_pl,
            ops['labels_pl']: batch_label,
            ops['is_training']: is_training,
            ops['mask_pl']: mask_padded.astype(float),
        }


        atts1,atts2,atts3, pred = sess.run(
            [ops['atts1'], ops['atts2'], ops['atts3'], 
             ops['pred']]
            ,feed_dict=feed_dict)         
        if len(y_pred)==0:
            y_pred= np.squeeze(pred)
        else:
            y_pred=np.concatenate((y_pred,pred),axis=0)
            
    with h5py.File(os.path.join(H5_OUT,'{0}.h5'.format(FLAGS.name)), "w") as fh5:
        dset = fh5.create_dataset("DNN", data=y_pred)
        dset = fh5.create_dataset("pid", data=current_label[:num_batches*(BATCH_SIZE)])


################################################          
    

if __name__=='__main__':
  eval()

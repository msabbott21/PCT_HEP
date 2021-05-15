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
import pct_best as MODEL
import tf_util
#import pct as MODEL

# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPUs to use [default: 0]')
parser.add_argument('--model_path', default='PU', help='Model checkpoint path')
parser.add_argument('--batch', type=int, default=64, help='Batch Size  during training [default: 64]')
parser.add_argument('--num_point', type=int, default=100, help='Point Number [default: 500]')
parser.add_argument('--data_dir', default='/pnfs/psi.ch/cms/trivcat/store/user/vmikuni/EMD_SF/', help='directory with data')
parser.add_argument('--nfeat', type=int, default=13, help='Number of features [default: 8]')
parser.add_argument('--ncat', type=int, default=2, help='Number of categories [default: 2]')
parser.add_argument('--name', default="", help='name of the output file')
parser.add_argument('--h5_folder', default="../h5/", help='folder to store output files')
parser.add_argument('--sample', default='qg', help='sample to use')
parser.add_argument('--simple', action='store_true', default=False,help='Use simplified model')
parser.add_argument('--num_bes', type=int, default=123, help='Number of BES variables  [default: 123]')
parser.add_argument('--decay_step', type=int, default=5000000, help='Decay step for lr decay [default: 5000000]')
parser.add_argument('--wd', type=float, default=0.0, help='Weight Decay [Default: 0.0]')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')

FLAGS = parser.parse_args()
MODEL_PATH = FLAGS.model_path
DATA_DIR = FLAGS.data_dir
H5_DIR = os.path.join(BASE_DIR, DATA_DIR)
H5_OUT = FLAGS.h5_folder
if not os.path.exists(H5_OUT): os.mkdir(H5_OUT)  

# MAIN SCRIPT
NUM_POINT = FLAGS.num_point
BATCH_SIZE = FLAGS.batch
NFEATURES = FLAGS.nfeat
SAMPLE = FLAGS.sample
NUM_BES = FLAGS.num_bes

NUM_CATEGORIES = FLAGS.ncat

DECAY_RATE = FLAGS.decay_rate
DECAY_STEP = FLAGS.decay_step
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99
BASE_LEARNING_RATE = FLAGS.learning_rate
LEARNING_RATE_CLIP = 1e-6
#Only used to get how many parts per category

print('#### Batch Size : {0}'.format(BATCH_SIZE))
print('#### Point Number: {0}'.format(NUM_POINT))
print('#### Using GPUs: {0}'.format(FLAGS.gpu))



    
print('### Starting evaluation')
multi=False
if SAMPLE == 'qg':
    EVALUATE_FILE = os.path.join(DATA_DIR, 'evaluate_PYTHIA.h5')
elif SAMPLE == 'top':    
    EVALUATE_FILE = os.path.join(DATA_DIR, 'test_ttbar.h5')
elif SAMPLE == 'multi':
    multi = True
    EVALUATE_FILE = os.path.join(DATA_DIR, 'eval_multi_100P_Jedi.h5')
elif SAMPLE == 'best':
    multi = True
    #EVALUATE_FILE = os.path.join(DATA_DIR, 'eval_multi_100P_Jedi.h5')
    EVALUATE_FILE = [os.path.join(DATA_DIR, mySamp+'Sample_2017_BESTinputs_test_flattened_standardized.h5') for mySamp in ["WW","ZZ","HH","TT","QCD","BB"]]
else:
    sys.exit("ERROR: SAMPLE NOT FOUND")

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, LEARNING_RATE_CLIP) # CLIP THE LEARNING RATE!
    return learning_rate 
  
def eval():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(FLAGS.gpu)):
            pointclouds_lab,pointclouds_higgs,pointclouds_bottom,pointclouds_top,pointclouds_W,pointclouds_Z, bes_vars,  mask_pl,  labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT,NFEATURES,NUM_BES)
            
            is_training = tf.placeholder(tf.bool, shape=())
            
            #batch = tf.Variable(0)
            batch = tf.Variable(0, trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            #if OPTIMIZER == 'momentum':
            #    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
#elif OPTIMIZER == 'adam':                
 #               optimizer = tf.train.AdamOptimizer(learning_rate)
  #          train_op = optimizer.minimize(loss, global_step=batch)
            
            if FLAGS.simple:                
                pred,atts1,atts2,atts3 = MODEL.get_model_simple(pointclouds_pl,mask_pl, is_training=is_training,scname='PL',num_class=NUM_CATEGORIES)           
            else:
                #pred,atts1,atts2,atts3 = MODEL.get_model(pointclouds_pl,mask_pl, is_training=is_training,scname='PL',num_class=NUM_CATEGORIES)
                pred_lab,att1,att2,att3 = MODEL.get_model(pointclouds_lab,mask_pl, 
                                                             is_training=is_training,
                                                             scname='LAB',
                                                             bn_decay=bn_decay, weight_decay=FLAGS.wd)

                pred_higgs,att1,att2,att3 = MODEL.get_model(pointclouds_higgs,mask_pl, 
                                                             is_training=is_training,
                                                             scname='HIGGS',
                                                             bn_decay=bn_decay, weight_decay=FLAGS.wd)
                pred_bottom,att1,att2,att3 = MODEL.get_model(pointclouds_bottom,mask_pl, 
                                                             is_training=is_training,
                                                             scname='BOTTOM',
                                                             bn_decay=bn_decay, weight_decay=FLAGS.wd)
                pred_top,att1,att2,att3 = MODEL.get_model(pointclouds_top,mask_pl, 
                                                             is_training=is_training,
                                                             scname='TOP',
                                                             bn_decay=bn_decay, weight_decay=FLAGS.wd)
                pred_W,att1,att2,att3 = MODEL.get_model(pointclouds_W,mask_pl, 
                                                             is_training=is_training,
                                                             scname='W',
                                                             bn_decay=bn_decay, weight_decay=FLAGS.wd)
                pred_Z,att1,att2,att3 = MODEL.get_model(pointclouds_Z,mask_pl, 
                                                             is_training=is_training,
                                                             scname='Z',
                                                             bn_decay=bn_decay, weight_decay=FLAGS.wd)

                bes_transform = tf_util.fully_connected(bes_vars, 128, bn=True, is_training=is_training,
                                                        activation_fn=tf.nn.relu,
                                                        scope='fc_bes',bn_decay=bn_decay)
                
                net = tf.concat([pred_lab,pred_higgs,pred_top,pred_W,pred_Z,bes_transform],-1)
                pred = MODEL.get_extractor(net,num_class=NUM_CATEGORIES,layer_size=256,
                                           is_training=is_training,
                                           scname='PRED',
                                           bn_decay=bn_decay, weight_decay=FLAGS.wd)
                
            loss = MODEL.get_loss(pred, labels_pl,NUM_CATEGORIES)
            pred=tf.nn.softmax(pred)            
            saver = tf.train.Saver()
          

    
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        merged = tf.summary.merge_all()

        saver.restore(sess,os.path.join(MODEL_PATH,'model.ckpt'))
        #saver.restore(sess,os.path.join('../logs',MODEL_PATH,'model.ckpt'))
        print('model restored')
        
        

        ops = {'pointclouds_lab': pointclouds_lab,
               'pointclouds_higgs': pointclouds_higgs,
               'pointclouds_bottom': pointclouds_bottom,
               'pointclouds_top': pointclouds_top,
               'pointclouds_W': pointclouds_W,
               'pointclouds_Z': pointclouds_Z,
               'bes_vars': bes_vars,
               #'pointclouds_pl': pointclouds_pl,
               
               'labels_pl': labels_pl,
               'mask_pl': mask_pl,
               'attention':att1,
               
               'is_training': is_training,
               'pred': pred,
               
               #'atts1': atts1,
               #'atts2': atts2,
               #'atts3': atts3,
               'loss': loss,
               #'train_op': train_op,
               'learning_rate':learning_rate,
               'merged': merged,
               'step': batch,
        }
            
        eval_one_epoch(sess,ops)

def get_batch(data,label, start_idx, end_idx):
    batch_label = label[start_idx:end_idx]
    #batch_data_pl = data_pl[start_idx:end_idx,:,:]
    batch_data = {}
    for dataset in data:
        batch_data[dataset] = data[dataset][start_idx:end_idx]
        
    return batch_data, batch_label

def eval_one_epoch(sess,ops):
    is_training = False
    y_pred = []
    
    #current_data_pl, current_label = provider.load_h5(EVALUATE_FILE,'class')
    #current_data_pl, current_label = provider.load_h5BEST(EVALUATE_FILE)
    data, label = provider.load_bes(EVALUATE_FILE) 
    if multi and not SAMPLE == 'best':
        label=np.argmax(label,axis=-1)
    #file_size = current_data_pl.shape[0]
    file_size = data['lab'].shape[0]
    num_batches = file_size // BATCH_SIZE        
    #num_batches = 4
    for batch_idx in range(num_batches):
        start_idx = batch_idx * (BATCH_SIZE)
        end_idx = (batch_idx+1) * (BATCH_SIZE)
        #batch_data_pl, batch_label = get_batch(current_data_pl, current_label,start_idx, end_idx)
        batch_data, batch_label = get_batch(data, label,start_idx, end_idx)
        mask_padded = batch_data['lab'][:,:,2]==0

        """
        feed_dict = {             
            ops['pointclouds_lab']: batch_data_lab,
            ops['pointclouds_higgs']: batch_data_higgs,
            ops['pointclouds_bottom']: batch_data_bottom,
            ops['pointclouds_top']: batch_data_top,
            ops['pointclouds_W']: batch_data_W,
            ops['pointclouds_Z']: batch_data_Z,
            ops['bes_vars']: batch_data_bes,
            
            ops['labels_pl']: batch_label,
            ops['is_training']: is_training,
            ops['mask_pl']: mask_padded.astype(float),
        }
        """
        feed_dict = {             
            ops['pointclouds_lab']: batch_data['lab'],
            ops['pointclouds_higgs']: batch_data['H'],
            ops['pointclouds_bottom']: batch_data['b'],
            ops['pointclouds_top']: batch_data['top'],
            ops['pointclouds_W']: batch_data['W'],
            ops['pointclouds_Z']: batch_data['Z'],
            ops['bes_vars']:batch_data['bes'],
            
            ops['labels_pl']: batch_label,
            ops['is_training']: is_training,
            ops['mask_pl']: mask_padded.astype(float),
        }


        #atts1,atts2,atts3, pred = sess.run(
        #    [ops['atts1'], ops['atts2'], ops['atts3'], 
        #     ops['pred']]
        #    ,feed_dict=feed_dict)
        summary, step, loss,pred,lr = sess.run([ops['merged'], ops['step'],
                                                ops['loss'],ops['pred'],
                                                ops['learning_rate']
                                         ],
                                            feed_dict=feed_dict)
        
        if len(y_pred)==0:
            y_pred= np.squeeze(pred)
        else:
            y_pred=np.concatenate((y_pred,pred),axis=0)
            
    with h5py.File(os.path.join(H5_OUT,'{0}.h5'.format(FLAGS.name)), "w") as fh5:
        dset = fh5.create_dataset("DNN", data=y_pred)
        dset = fh5.create_dataset("pid", data=label[:num_batches*(BATCH_SIZE)])

def eval_one_epoch_3(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    loss_sum = 0
    y_source=[]

    data, label = provider.load_bes(TRAIN_FILE)    
    if multi:
        label=np.argmax(label,axis=-1)
        
    label, idx = provider.shuffle_bes(np.squeeze(label))
    data = shuffle_idx(idx,data)

    file_size = data['lab'].shape[0]
    num_batches = file_size // (BATCH_SIZE)
    if FLAGS.test:
        num_batches = 4
        
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    for batch_idx in range(num_batches):
            
        start_idx = batch_idx * (BATCH_SIZE)
        end_idx = (batch_idx+1) * (BATCH_SIZE)
        batch_data, batch_label = get_batch(data, label,start_idx, end_idx)
        mask_padded = batch_data['lab'][:,:,2]==0 
        
        feed_dict = {             
            ops['pointclouds_lab']: batch_data['lab'],
            ops['pointclouds_higgs']: batch_data['H'],
            ops['pointclouds_bottom']: batch_data['b'],
            ops['pointclouds_top']: batch_data['top'],
            ops['pointclouds_W']: batch_data['W'],
            ops['pointclouds_Z']: batch_data['Z'],
            ops['bes_vars']:batch_data['bes'],
            
            ops['labels_pl']: batch_label,
            ops['is_training']: is_training,
            ops['mask_pl']: mask_padded.astype(float),
        }
            
        if batch_idx ==0:
            start_time = time.time()
            
        summary, step, loss,pred,lr = sess.run([ops['merged'], ops['step'],
                                                ops['loss'],ops['pred'],
                                                ops['learning_rate']
                                         ],
                                            feed_dict=feed_dict)
        

        if batch_idx ==0:
            duration = time.time() - start_time
            log_string("Eval time: "+str(duration)) 
            log_string("Learning rate: "+str(lr)) 
            #log_string("{}".format(sub_feat))


        test_writer.add_summary(summary, step)
           
            
        loss_sum += np.mean(loss)                                        
        if len(y_source)==0:
            y_source = np.squeeze(pred)
        else:
            y_source=np.concatenate((y_source,np.squeeze(pred)),axis=0)
            

    if multi:
        name_convert = {
            0:'Gluon',
            1:'Quark',
            2:'Z',
            3:'W',
            4:'Top',
            
        }
        label = label[:num_batches*(BATCH_SIZE)]
        for isample in np.unique(label):            
            fpr, tpr, _ = metrics.roc_curve(label==isample, y_source[:,isample], pos_label=1)    
            log_string("Class: {}, AUC: {}".format(name_convert[isample],metrics.auc(fpr, tpr)))
            bineff = np.argmax(fpr>0.1)
            log_string('SOURCE: effS at {0} effB = {1}'.format(tpr[bineff],fpr[bineff]))
        log_string('mean loss: %f' % (loss_sum*1.0 / float(num_batches)))
    else:
        fpr, tpr, _ = metrics.roc_curve(label[:num_batches*(BATCH_SIZE)], y_source[:,1], pos_label=1)    
        log_string("AUC: {}".format(metrics.auc(fpr, tpr)))

        bineff = np.argmax(tpr>0.3)

        log_string('SOURCE: 1/effB at {0} effS = {1}'.format(tpr[bineff],1.0/fpr[bineff]))
        log_string('mean loss: %f' % (loss_sum*1.0 / float(num_batches)))
    EPOCH_CNT += 1


    return loss_sum*1.0 / float(num_batches)
    



################################################          
    

if __name__=='__main__':
  eval()

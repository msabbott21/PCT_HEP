from enum import unique
import os
import sys
import numpy as np
import h5py
import tensorflow as tf
import matplotlib
matplotlib.use('Agg') #prevents opening displays, must use before pyplot
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
            data: B,N,... numpy array
            label: B,N, numpy array
        Return:
            shuffled data, label and shuffle indices
    """
    #np.random.seed(0)
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    for dataset in data:
        data[dataset] = data[dataset][idx]
    return data, labels[idx]
    # return data, labels[idx], idx



def rotate_point_cloud(batch_data):
  """ Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
  """
  rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
  for k in xrange(batch_data.shape[0]):
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                  [0, 1, 0],
                  [-sinval, 0, cosval]])
    shape_pc = batch_data[k, ...]
    rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
  return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
  """ Rotate the point cloud along up direction with certain angle.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
  """
  rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
  for k in xrange(batch_data.shape[0]):
    #rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                  [0, 1, 0],
                  [-sinval, 0, cosval]])
    shape_pc = batch_data[k, ...]
    rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
  return rotated_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
  """ Randomly perturb the point clouds by small rotations
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
  """
  rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
  for k in xrange(batch_data.shape[0]):
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
             [0,np.cos(angles[0]),-np.sin(angles[0])],
             [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
             [0,1,0],
             [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
             [np.sin(angles[2]),np.cos(angles[2]),0],
             [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    shape_pc = batch_data[k, ...]
    rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
  return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
  """ Randomly jitter points. jittering is per point.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, jittered batch of point clouds
  """
  B, N, C = batch_data.shape
  assert(clip > 0)
  jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
  jittered_data += batch_data
  return jittered_data

def shift_point_cloud(batch_data, shift_range=0.1):
  """ Randomly shift point cloud. Shift is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, shifted batch of point clouds
  """
  B, N, C = batch_data.shape
  shifts = np.random.uniform(-shift_range, shift_range, (B,3))
  for batch_index in range(B):
    batch_data[batch_index,:,:] += shifts[batch_index,:]
  return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
  """ Randomly scale the point cloud. Scale is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, scaled batch of point clouds
  """
  B, N, C = batch_data.shape
  scales = np.random.uniform(scale_low, scale_high, B)
  for batch_index in range(B):
    batch_data[batch_index,:,:] *= scales[batch_index]
  return batch_data

def getDataFiles(list_filename):
  return [line.rstrip() for line in open(list_filename)]

def getNevents(h5_filename):
    return h5py.File(h5_filename, 'r')["BES_vars"].shape[0]

def getMinEvents(h5_filenames):
    return np.min([getNevents(file) for file in h5_filenames])
    # events = [getNevents(file) for file in h5_filenames]
    # return np.min(events)
    # minEvents = np.inf
    # for file in h5_filenames: 
    #     events = getNevents(file)
    #     if events < minEvents: minEvents = events 
    # return minEvents

def randomize_sub_epochs(sub_epoch_shifts, is_init = False):
    """ Shuffle Sub Epoch order.
        This randomizes the order of events.
    """
    #np.random.seed(0)
    if is_init:
        for i in range(len(sub_epoch_shifts[0])):
            np.random.shuffle(sub_epoch_shifts[0][i])
            np.random.shuffle(sub_epoch_shifts[1][i])
    else:
        # for i in range(len(sub_epoch_shifts)):
        np.random.shuffle(sub_epoch_shifts)
    return sub_epoch_shifts



def getFrames(log_dir, justDep = False):

    # List of every Dependent Frame:
    dep_frames =  [ "PF_cands_WFrame", "PF_cands_ZFrame", "PF_cands_HiggsFrame", "PF_cands_TopFrame",
                        "PF_cands_BottomFrame", "PF_cands_LabFrame", "PF_cands_ak8Frame", "PF_cands_ak8SoftDropFrame"
                        "PF_cands_50GeVFrame", "PF_cands_100GeVFrame", "PF_cands_150GeVFrame", "PF_cands_200GeVFrame",
                        "PF_cands_250GeVFrame", "PF_cands_300GeVFrame", "PF_cands_350GeVFrame", "PF_cands_400GeVFrame"]

    if justDep: return dep_frames

    frameKeys = log_dir
    frameKeys = frameKeys[frameKeys.find('_')+1:] # trim events+S_ (100kS_LV -> LV)
    frames = []
    for frameKey in frameKeys: # Iterate over each character, get frames
        if   frameKey == "L": frames.append("PF_cands_LabFrame")
        elif frameKey == "V": frames.append("SV_vars")
        # elif frameKey == "E": frames = ["BES_vars"] + frames
        elif frameKey == "E": continue
        elif frameKey == "W": frames.append("PF_cands_WFrame")
        elif frameKey == "Z": frames.append("PF_cands_ZFrame")
        elif frameKey == "H": frames.append("PF_cands_HiggsFrame")
        elif frameKey == "T": frames.append("PF_cands_TopFrame")
        elif frameKey == "B": frames.append("PF_cands_BottomFrame")
        elif frameKey == "K": frames.append("PF_cands_ak8Frame")
        elif frameKey == "D": frames.append("PF_cands_ak8SoftDropFrame")
        elif frameKey == "1": frames.append("PF_cands_50GeVFrame")
        elif frameKey == "2": frames.append("PF_cands_100GeVFrame")
        elif frameKey == "3": frames.append("PF_cands_150GeVFrame")
        elif frameKey == "4": frames.append("PF_cands_200GeVFrame")
        elif frameKey == "5": frames.append("PF_cands_250GeVFrame")
        elif frameKey == "6": frames.append("PF_cands_300GeVFrame")
        elif frameKey == "7": frames.append("PF_cands_350GeVFrame")
        elif frameKey == "8": frames.append("PF_cands_400GeVFrame")
        else:
            print("FRAME ERROR: Invalid frame: " + frameKey)
            quit()        

    frames.sort()
    # # Make sure that BES_vars is always the first frame
    # if "E" in frameKeys: frames = ["BES_vars"] + frames

    # Make sure that BES_vars is always the last frame
    if "E" in frameKeys: frames = frames + ["BES_vars"] 

    return frames, dep_frames


def load_h5BEST(h5_filenames, frames, events, is_training=False, mask=[]):
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

    dep_frames = getFrames("", justDep=True)

    # print("Loading Data: ", frames, events)

    # Grab h5 Files
    fs = [h5py.File(h5_filename,'r') for h5_filename in h5_filenames]

    # Initialize data dictionary
    data = dict()
    events = list(events)
    # Load the Invariant Frame Data:
    # Frame invariant shape: N_events x N_PFCands(50) x N_vars(9)
    # h5 Var indices: 0-PUPPI_Weights, 1-abs(pdgID), 2-charge, 3-isChargedHadron, 4-isElectron, 5-isMuon, 6-isNeutralHadron, 7-isPhoton, 8-pdgID
    data["PF_cands_AllFrame"] = [ f["PF_cands_AllFrame"][events[i]][..., [2,4,5,3,6,7] ] for i, f in enumerate(fs) ]
    # ^This data structure is BATCH_SIZE x N_PFCands x [charge, isElectron, isMuon, isChargedHadron, isNeutralHadron, isPhoton](6)

    # Create the truth labels:
    label = np.concatenate( [np.full(len(data["PF_cands_AllFrame"][i]),i) for i in range(len(fs))] )

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

    for frame in frames:
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
            h5Dir = "/uscms/home/bonillaj/nobackup/h5samples_ULv1/" 
            if is_training: h5suffix = "Sample_2017_BESTinputs_train_flattened_newBEST_Basic.h5"
            else:           h5suffix = "Sample_2017_BESTinputs_validation_flattened_newBEST_Basic.h5"


            data[frame] = np.concatenate( [h5py.File(h5Dir+samp+h5suffix,'r')[frame][events[i]][:,mask] 
                                            for i, samp in enumerate(["WW","ZZ","HH","TT","BB","QCD"])] )

        elif frame in dep_frames:
            # Frame dependent shape: N_events x N_PFCands(50) x N_vars(11)
            # h5 Var indices: 0-deltaEta, 1-deltaPhi, 2-deltaR, 3-energy, 4-logEnergy, 5-logEnergyRatio, 6-logpT, 7-logpTRatio, 8-px, 9-py, 10-pz
            
            # This data structure is BATCH_SIZE x N_PFCands x [deltaEta, deltaPhi, deltaR, logEnergyRatio, logpTRatio, logpT, logEnergy],
            #  concatenated with the Invariant variables, then concatenated across the h5 samples
            data[frame] = np.concatenate( [ np.concatenate( (f[frame][events[i]][..., [0,1,2,5,7,6,4] ], 
                                            data["PF_cands_AllFrame"][i]), axis=2 ) for i, f in enumerate(fs) ] )

        else:
            print("FRAME ERROR: Invalid frame: " + frame)
            quit()

    del data["PF_cands_AllFrame"] # delete this now that we are done

    # print("data", np.array(data).shape, np.array(data[0]).shape, np.array(data[0][0]).shape, np.array(data[0][0][0]).shape)

    return data, label

###### The following code is for the tensorflow 2 implementation:

# def convertData(data, labels, frames):
#         dat = []
#         for frame in frames: dat.append(data[frame])
#         dat = tuple(dat)

#         dat = (data[frame] for frame in frames)
#         labs = tf.one_hot(labels, 6)

#         unique_labels = [0,1,2,3,4,5]
#         truth = np.zeros(( len(labels), len(unique_labels) )) # shape: N,6 filled w/ 0s
#         for ev, lab in enumerate(labels):
#             truth[ev,lab] = 1.
#         return dat, truth

def subEpochGenerator(files, bframes, batch_size, allEvents, mask=[]):

    frames = [str(bframe, 'utf-8') for bframe in bframes]
    batches = len(allEvents[0]) // batch_size

    for batch in range(batches):
        start = batch * batch_size
        end = (batch+1) * batch_size

        events = []
        for i in range(len(allEvents)):
            # tempEvents = np.array(allEvents[i][start:end])
            # tempEvents.sort()
            # events.append(tempEvents)
            values, counts = np.unique(allEvents[i][start:end], return_counts=True)
            tempEvents = []
            for i, value in enumerate(values):
                for j in range(counts[i]): tempEvents.append(value + j)
            events.append(tempEvents)
            # if np.any(counts[:] > 1):
            #     tempEvents = []
            #     prevVal = None
            #     # for value in values:
            #     #     if value == prevVal: tempEvents.append(value + 1)
            #     #     else:                tempEvents.append(value)
            #     #     prevVal = value
            # else: tempEvents = values
            # events.append(values)

        
        # events = [ allEvents[i][start:end] for i in range(len(allEvents)) ]

        data, labels = load_h5BEST(files, frames, events, mask=mask)

        yield tuple([data[frame] for frame in frames]), tf.one_hot(labels, 6)

def load_subEpoch(files, frames, batch_size, num_bes, allEvents, mask=[]):  

    buff = batch_size * len(allEvents)

    if ("BES_vars" in frames):
        dataset = tf.data.Dataset.from_generator(
                subEpochGenerator, args=[files, frames, batch_size, allEvents, mask],
                output_signature=(
                    tuple( [tf.TensorSpec(shape=(buff, 50, 13), dtype=tf.float64) for i in range(len(frames) - 1)]
                            + [tf.TensorSpec(shape= (buff, num_bes), dtype=tf.float64)] ), 
                    tf.TensorSpec(shape=(buff, 6), dtype=tf.int32)  )
                )
    else:
        dataset = tf.data.Dataset.from_generator(
                subEpochGenerator, args=[files, frames, batch_size, allEvents],
                output_signature=(
                    tuple( [tf.TensorSpec(shape=(buff, 50, 13), dtype=tf.float64) for i in range(len(frames))] ), 
                    tf.TensorSpec(shape=(buff, 6),  dtype=tf.int32)  )
                     )

    return dataset


def dataGenerator(files, bframes, batch_size, is_training, mask=[]):
    # need to make a generator.
    # this generator will load a random batch of data.
    # will do something like the SUB EPOCH SHIFTS in tf1 code.
    # do np.arange for each h5 file, shuffle, iterate over all events
    # generator yields what loadh5BEST would return for a batch of random indices 

    idx = [np.arange(getNevents(file)) for file in files]
    batches = getMinEvents(files) // batch_size

    frames = [str(bframe, 'utf-8') for bframe in bframes]
    for ind in idx: np.random.shuffle(ind)

    for batch in range(batches):
        # print("\nBATCH: " + str(batch) + " is_train: " + str(is_training) + "\n")
        # print(np.array(idx[0]).shape)
        # print(idx[0][:10])
        start = batch * batch_size
        end = (batch+1) * batch_size
        # if batch + 1 == batches: end = None
        # else:                    end = (batch+1) * batch_size

        events = []
        for inds in idx:
            ev = inds[start:end]
            ev.sort()
            events.append(list(ev))
        
        data, labels = load_h5BEST(files, frames, events, is_training, mask=mask)
        # dat = [data[frame] for frame in frames]
        # print(len(dat),dat[0].shape)
        # yield dat, tf.one_hot(labels, 6)

        yield tuple([data[frame] for frame in frames]), tf.one_hot(labels, 6)


# def load_bes(h5_filenames, frames, nevts=-1,batch_size=64):  
def load_bes(h5_filenames, frames, batch_size, num_bes, is_training = False, mask=[]):  

    # tf will call generator again when all events are exhausted
    # needs to work for train and eval
    # last batch should grab remaining events for all files
    # this gen is passed to to dataset.from_generator

    buff = batch_size * 6
    if ("BES_vars" in frames):
        dataset = tf.data.Dataset.from_generator(
                dataGenerator, args=[h5_filenames, frames, batch_size, is_training, mask],
                output_signature=(
                    tuple( [tf.TensorSpec(shape=(buff, 50, 13), dtype=tf.float64) for i in range(len(frames) - 1)]
                            + [tf.TensorSpec(shape= (buff, num_bes), dtype=tf.float64)] ), 
                                    tf.TensorSpec(shape=(buff, 6),      dtype=tf.int32)  )
                )
    else:
        dataset = tf.data.Dataset.from_generator(
                dataGenerator, args=[h5_filenames, frames, batch_size, is_training],
                output_signature=(
                    tuple( [tf.TensorSpec(shape=(buff, 50, 13), dtype=tf.float64) for i in range(len(frames))] ), 
                    tf.TensorSpec(shape=(buff, 6),      dtype=tf.int32)  )
                     )

    # print(dataset)
    # return dataset.repeat()
    if is_training: return dataset.shuffle(buff, reshuffle_each_iteration=True).repeat()
    else:           return dataset.repeat()
    # return tf_data

    # fs  = [h5py.File(h5_filename,'r') for h5_filename in h5_filenames]
    # idx = [getNevents(h5_filename) for h5_filename in h5_filenames]

    # nevts=int(nevts)
    # if nevts == -1:
    #     nevts =f['BES_vars'].shape[0]  
    
    # data=(f['BES_vars'][:nevts],
    #       f['LabFrame_PFcands'][:nevts],
    #       f['HiggsFrame_PFcands'][:nevts],
    #       f['BottomFrame_PFcands'][:nevts],
    #       f['TopFrame_PFcands'][:nevts],
    #       f['WFrame_PFcands'][:nevts],
    #       f['ZFrame_PFcands'][:nevts])

    #Will now convert to a tf dataset. should be more efficient than the previous strategy
    #Shuffling and batching is also automatic, so no need to shuffle again later

    # dataset =tf.data.Dataset.from_tensor_slices(data)
    # #label = f['pid'][:nevts].astype(int) #No stored in the tet file, will use a dummy instead
    # label = np.random.randint(1, size=(nevts,2))
    # dataset_label = tf.data.Dataset.from_tensor_slices(label)
    # tf_data = tf.data.Dataset.zip((dataset, dataset_label)).shuffle(nevts).batch(batch_size)

    # return tf_data


def plotPerformance(lossList, accList, suffix, plotDir): 
    loss = lossList[0]
    val_loss = lossList[1]
    acc = accList[0]
    val_acc = accList[1]

    # plot loss vs epoch
    plt.figure()
    plt.plot(loss, label='loss; Min loss: ' + str(np.min(loss))[:6] + ', Epoch: ' + str(np.argmin(loss)) )
    plt.plot(val_loss, label='val_loss; Min val_loss: ' + str(np.min(val_loss))[:6] + ', Epoch: ' + str(np.argmin(val_loss)) )
    plt.title(suffix + " loss and val_loss vs. epochs")
    plt.legend(loc="upper right")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    if not os.path.isdir(plotDir): os.makedirs(plotDir)
    # plt.savefig(plotDir+suffix+"_loss.pdf")
    plt.savefig(plotDir+suffix+"_loss.png")
    plt.close()
 

    # plot accuracy vs epoch
    plt.figure()
    plt.plot(acc,     label='acc; Max acc: '  + str(np.max(acc))[:6] + ', Epoch: ' + str(np.argmax(acc)) )
    plt.plot(val_acc, label='val_acc; Max val_acc: ' + str(np.max(val_acc))[:6] + ', Epoch: ' + str(np.argmax(val_acc)) )
    plt.title(suffix + " acc and val_acc vs. epochs")
    plt.legend(loc="lower right")
    plt.xlabel('epoch')
    plt.ylabel('acc')
    # plt.savefig(plotDir+suffix+"_acc.pdf")
    plt.savefig(plotDir+suffix+"_acc.png")
    plt.close()










"""
def load_h5BEST(h5_filenames, num_points, events):
    # print(begin,end)
    # print("load_h5BEST",num_points)
    # print("Loading Data: ", begin, end)
    fs = [h5py.File(h5_filename,'r') for h5_filename in h5_filenames]

    # Three types of h5 data structures; Frame Invariant, Frame Dependent, and Secondary Vertex
    # First three PCT input variables NEED to be deltaEta, then deltaPhi (for the knn), then deltaR
    # Third PCT input variable is used for 'mask_padded'. Choose something that is NEVER 0 when it is filled. (deltaR, SV_nTracks) 
    # Frame Invariant vars (isMuon, charge, etc.) will be added to each Frame Dependent PCT input array; 
    # 13 vars total:
    # [deltaEta, deltaPhi, deltaR, logEnergyRatio, logpTRatio, logpT, logEnergy, charge, isElectron, isMuon, isChargedHadron, isNeutralHadron, isPhoton]
    # Secondary Vertex vars will be a seperate "frame"; Will need to 'scale' energy/mass and calculate delta;
    # 9 vars total:
    # [deltaEta, deltaPhi, nTracks, Ndof, chi2, logMassRatio, logpTRatio, logMass, logpT ]

    # Frame invariant shape: N_events x N_PFCands(50) x N_vars(9)
    # h5 Var indices: 0-PUPPI_Weights, 1-abs(pdgID), 2-charge, 3-isChargedHadron, 4-isElectron, 5-isMuon, 6-isNeutralHadron, 7-isPhoton, 8-pdgID
    # data_inv = [ f["PF_cands_AllFrame"][:nevts,0:num_points][...,[2,4,5,3,6,7]  ] for f in fs ]
    # data_inv = [ f["PF_cands_AllFrame"][begin:end,0:num_points][...,[2,4,5,3,6,7]  ] for f in fs ]
    # data_inv = [ f["PF_cands_AllFrame"][begin[i]:end[i],0:num_points][...,[2,4,5,3,6,7]  ] for i, f in enumerate(fs) ]
    data_inv = [ f["PF_cands_AllFrame"][events[i],0:num_points][...,[2,4,5,3,6,7]  ] for i, f in enumerate(fs) ]
    # ^This data structure is N_events x N_PFCands x [charge, isElectron, isMuon, isChargedHadron, isNeutralHadron, isPhoton]

    # Frame dependent shape: N_events x N_PFCands(50) x N_vars(11)
    # h5 Var indices: 0-deltaEta, 1-deltaPhi, 2-deltaR, 3-energy, 4-logEnergy, 5-logEnergyRatio, 6-logpT, 7-logpTRatio, 8-px, 9-py, 10-pz
    # data_lab = [ f["PF_cands_LabFrame"][:nevts,0:num_points][...,[0,1,2,5,7,6,4]] for f in fs ]
    # data_lab = [ f["PF_cands_LabFrame"][begin:end,0:num_points][...,[0,1,2,5,7,6,4]] for f in fs ]
    # data_lab = [ f["PF_cands_LabFrame"][begin[i]:end[i],0:num_points][...,[0,1,2,5,7,6,4]] for i, f in enumerate(fs) ]
    data_lab = [ f["PF_cands_LabFrame"][events[i],0:num_points][...,[0,1,2,5,7,6,4]] for i, f in enumerate(fs) ]
    # ^This data structure is N_events x N_PFCands x [deltaEta, deltaPhi, deltaR, logEnergyRatio, logpTRatio, logpT, logEnergy]

    data = [ np.concatenate( (data_lab[i], data_inv[i]), axis=2 ) for i in range(0,len(fs)) ]
    # print("data", np.array(data).shape, np.array(data[0]).shape, np.array(data[0][0]).shape, np.array(data[0][0][0]).shape)
    label = [np.full(len(data[i]),i) for i in range(0,len(data))]

    return (np.concatenate(data), np.concatenate(label))

    # Secondary Vertex shape: N_events x N_SV(10) x N_vars(7)
    # h5 Var indices: 0-SV_Ndof, 1-SV_chi2, 2-SV_eta, 3-SV_mass, 4-SV_nTracks, 5-SV_phi, 6-SV_pt
    data_sv = [f["SV_vars"][begin:end,:,:] for f in fs]
    data_sv_etaPhi = [ arr[:,:][...,[2,5]] for arr in data_sv] # Use this to calc delta eta/ delta phi
    with np.errstate(divide = 'ignore'): # The unfilled entries will raise a warning (log(0) = -inf). We mask them later, so ignore the warning.
        data_sv_logPtMass = [ np.log(arr[:,:][...,[3,6]]) for arr in data_sv] # Use this to calc log energy/mass ratios 
    data_sv_remain = [ arr[:,:][...,[4,0,1]] for arr in data_sv] # Remaining vars; Use nTracks for padded mask, so it is first.
    del data_sv

    # Need to create deltaEta, deltaPhi, logMassRatio, logpTRatio
    # Load jetAK8 data; indices: 545-jetAK8_eta, 546-jetAK8_mass, 547-jetAK8_phi, 548-jetAK8_pt
    data_jet = [ f["BES_vars"][begin:end][...,[545,546,547,548]] for f in fs ]
    data_sv_logRatios = []
    for i, arr in enumerate(data_jet): 

        data_sv_etaPhi[i] -= arr[:][...,np.newaxis,[0,2]] 
        data_sv_logRatios.append(data_sv_logPtMass[i] - np.log(arr[:][...,np.newaxis,[1,3]]))
    del data_jet

    data_sv = [ np.concatenate( (data_sv_etaPhi[i], data_sv_remain[i], 
                        data_sv_logRatios[i], data_sv_logPtMass[i]), axis=2 ) for i in range(0,len(fs)) ]
    del data_sv_etaPhi; del data_sv_remain; del data_sv_logRatios; del data_sv_logPtMass
    # print("data", np.array(data_sv).shape, np.array(data_sv[0]).shape, np.array(data_sv[0][0]).shape, np.array(data_sv[0][0][0]).shape)

    # inv = "PF_cands_AllFrame" 
    # This data structure is N_events x N_PFCands x [charge, isElectron, isMuon, isChargedHadron, isNeutralHadron, isPhoton]
    # data_inv = [ f[invPFKey][:nevts,0:num_points][...,[2,4,5,3,6,7]]   for f in fs ]

    # lab = "PF_cands_LabFrame" 
    # This data structure is N_events x N_PFCands x [deltaEta, deltaPhi, deltaR, logEnergyRatio, logpTRatio, logpT, logEnergy]
    # data_lab = [ f[labPFKey][:nevts,0:num_points][...,[0,1,2,5,7,6,4]] for f in fs ]

    # Secondary Vertex shape: N_events x N_SV(10) x N_vars(7)
    # h5 Var indices: 0-SV_Ndof, 1-SV_chi2, 2-SV_eta, 3-SV_mass, 4-SV_nTracks, 5-SV_phi, 6-SV_pt
    # sv = "SV_vars"

    # add SV stuff. make delta eta and delta phi. log mass log pt ratio as well (softdrop?). throw in the rest


    data = [ np.concatenate( (data_lab[i], data_inv[i]), axis=2 ) for i in range(0,len(fs)) ]
    # for i, arr in enumerate(data): print(arr.shape)
    # del data_inv; del data_lab
    # for i, arr in enumerate(data): print(arr.shape)
    # quit()
    # print("data")
    # print("data", np.array(data).shape, np.array(data[0]).shape, np.array(data[0][0]).shape, np.array(data[0][0][0]).shape)

    # print("Make label")
    label = [np.full(len(data[i]),i) for i in range(0,len(data))]
    # returnLabel = np.concatenate(label)

    return (np.concatenate(data), np.concatenate(label))
    # if debug: return (data, label)
    # else:     return (np.concatenate(data), np.concatenate(label))
    # return (np.concatenate(data), returnLabel)
"""

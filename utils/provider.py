import os
import sys
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler, MinMaxScaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification


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

  return data[idx,:], labels[idx],idx

def shuffle_bes(labels):
  """ Shuffle data and labels.
    Input:
      data: B,N,... numpy array
      label: B,N, numpy array
    Return:
      shuffled data, label and shuffle indices
  """
  np.random.seed(0)
  idx = np.arange(len(labels))
  np.random.shuffle(idx)
  return labels[idx],idx


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

def load_h5(h5_filename,mode='seg',unsup=False,glob=False,nevts=-1):
  global_pl = []
  f = h5py.File(h5_filename,'r')
  nevts=int(nevts)
  data = f['data'][:nevts]
  
  if mode == 'class':
    label = f['pid'][:nevts].astype(int)
  elif mode == 'seg':
    label = f['label'][:nevts].astype(int)
  else:
    print('No mode found')
  if glob:
    global_pl = f['global'][:nevts]
    return (data, label,global_pl)

  print("loaded {0} events".format(len(data)))
  return (data, label)

def load_bes(h5_filenames,nevts=-1):
  #print("load_h5BEST")
  fs = [h5py.File(h5_filename,'r') for h5_filename in h5_filenames]
  nevts=int(nevts)

  """
  #print("Make data") # The indices are 0-deltaEta, 9-deltaPhi, 6-pdgid, 7-charge, 2-px, 3-py, 5-pz, 1-e
  data_knn = [f['LabFrame_PFcands'][:nevts][...,[0,9]] for f in fs] # deltaEta and deltaPhi
  #print("data_knn",np.array(data_knn).shape, np.array(data_knn[0]).shape, np.array(data_knn[0][0]).shape, np.array(data_knn[0][0][0]).shape)
  data_deltaR = [np.sqrt(f['LabFrame_PFcands'][:nevts][...,0]**2 + f['LabFrame_PFcands'][:nevts][...,9]**2) for f in fs] # sqrt(deltaEta^2+deltaPhi^2)
  #print("data_deltaR", np.array(data_deltaR).shape, np.array(data_deltaR[0]).shape, np.array(data_deltaR[0][0]).shape, np.array(data_deltaR[0][0][0]).shape)
  data_logRelativeE = []
  data_logRelativePt = []
  data_logE = []
  data_logPt = []
  for f in fs:
    data_f_logRelativeE = np.zeros(np.array(f['LabFrame_PFcands'][:nevts][...,1]).shape)
    data_f_logRelativePt = np.zeros(data_f_logRelativeE.shape)
    data_f_logE = np.zeros(np.array(f['LabFrame_PFcands'][:nevts][...,1]).shape)
    data_f_logPt = np.zeros(data_f_logE.shape)
    for event in range(nevts):
      pfTempRelE = np.zeros(100)
      pfTempRelPt = np.zeros(100)
      pfTempE = np.zeros(100)
      pfTempPt = np.zeros(100)
      for pfCand in range(len(pfTemp)):
        if f['LabFrame_PFcands'][event][pfCand][1] > 0:
          pfTempRelE[pfCand] = np.log(np.divide(f['LabFrame_PFcands'][event][pfCand][1],f['BES_vars'][event][23]))
          pfTempE[pfCand] = np.log(f['LabFrame_PFcands'][event][pfCand][1])
        if f['LabFrame_PFcands'][event][pfCand][2]+f['LabFrame_PFcands'][event][pfCand][3] > 0:
          pfTempRelPt[pfCand] = np.log(np.divide(np.sqrt(f['LabFrame_PFcands'][event][pfCand][2]**2 + f['LabFrame_PFcands'][event][pfCand][3]**2),f['BES_vars'][event][53]))
          pfTempPt[pfCand] = np.log(np.sqrt(f['LabFrame_PFcands'][event][pfCand][2]**2 + f['LabFrame_PFcands'][event][pfCand][3]**2))
      data_f_logRelativeE[event]=pfTempRelE
      data_f_logRelativePt[event]=pfTempRelPt
      data_f_logE[event]=pfTempE
      data_f_logPt[event]=pfTempPt
    data_logRelativeE.append(data_f_logRelativeE)
    data_logRelativePt.append(data_f_logRelativePt)
    data_logE.append(data_f_logE)
    data_logPt.append(data_f_logPt)
  #print("data_logRelativeE", np.array(data_logRelativeE).shape, np.array(data_logRelativeE[0]).shape, np.array(data_logRelativeE[0][0]).shape, np.array(data_logRelativeE[0][0][0]).shape)
  #print("data_logRelativePt", np.array(data_logRelativePt).shape, np.array(data_logRelativePt[0]).shape, np.array(data_logRelativePt[0][0]).shape, np.array(data_logRelativePt[0][0][0]).shape)
  #print("data_logE", np.array(data_logE).shape, np.array(data_logE[0]).shape, np.array(data_logE[0][0]).shape, np.array(data_logE[0][0][0]).shape)
  #print("data_logPt", np.array(data_logPt).shape, np.array(data_logPt[0]).shape, np.array(data_logPt[0][0]).shape, np.array(data_logPt[0][0][0]).shape)
  #data_logRelativeE = [np.log((np.divide(f['LabFrame_PFcands'][:nevts][...,1], np.expand_dims(f['BES_vars'][:nevts,23],axis=1)))) for f in fs] # jetAK8_e is index 23
  #data_logRelativePt = [np.log((np.sqrt((f['LabFrame_PFcands'][:nevts][...,2]**2 + f['LabFrame_PFcands'][:nevts][...,3]**2))/f['BES_vars'][:nevts][...,53])) for f in fs] # jetAK8_pt is index 53
  #data_logMomenta = [numpy.log(f['LabFrame_PFcands'][:nevts][...,[2,3,5]]) for f in fs] # px, py, pz
  #data_logPt = [np.log(np.sqrt((f['LabFrame_PFcands'][:nevts][...,2]**2 + f['LabFrame_PFcands'][:nevts][...,3]**2))) for f in fs] # px, py
  #data_logE = [np.log(f['LabFrame_PFcands'][:nevts][...,1]) for f in fs] # e
  
  data_isElectron = [abs(f['LabFrame_PFcands'][:nevts][...,6])==11 for f in fs]
  #print("data_isElectron", np.array(data_isElectron).shape, np.array(data_isElectron[0]).shape, np.array(data_isElectron[0][0]).shape, np.array(data_isElectron[0][0][0]).shape)
  data_isMuon = [abs(f['LabFrame_PFcands'][:nevts][...,6])==13 for f in fs]
  #print("data_isMuon", np.array(data_isMuon).shape, np.array(data_isMuon[0]).shape, np.array(data_isMuon[0][0]).shape, np.array(data_isMuon[0][0][0]).shape)
  data_isChargedHadron = [(abs(f['LabFrame_PFcands'][:nevts][...,6])==211) | (abs(f['LabFrame_PFcands'][:nevts][...,6])==321) for f in fs]
  #print("data_isChargedHadron", np.array(data_isChargedHadron).shape, np.array(data_isChargedHadron[0]).shape, np.array(data_isChargedHadron[0][0]).shape, np.array(data_isChargedHadron[0][0][0]).shape)
  data_isNeutralHadron = [(abs(f['LabFrame_PFcands'][:nevts][...,6])==111) | (abs(f['LabFrame_PFcands'][:nevts][...,6])==130) | (abs(f['LabFrame_PFcands'][:nevts][...,6])==310) | (abs(f['LabFrame_PFcands'][:nevts][...,6])==311) for f in fs]
  #print("data_isNeutralHadron", np.array(data_isNeutralHadron).shape, np.array(data_isNeutralHadron[0]).shape, np.array(data_isNeutralHadron[0][0]).shape, np.array(data_isNeutralHadron[0][0][0]).shape)
  data_isPhoton = [abs(f['LabFrame_PFcands'][:nevts][...,6])==22 for f in fs]
  #print("data_isPhoton", np.array(data_isPhoton).shape, np.array(data_isPhoton[0]).shape, np.array(data_isPhoton[0][0]).shape, np.array(data_isPhoton[0][0][0]).shape)
  data_charge = [f['LabFrame_PFcands'][:nevts][...,7] for f in fs]
  #print("data_charge", np.array(data_charge).shape, np.array(data_charge[0]).shape, np.array(data_charge[0][0]).shape, np.array(data_charge[0][0][0]).shape)
  data = [np.concatenate((data_knn[i], np.expand_dims(data_deltaR[i],axis=2), np.expand_dims(data_logRelativeE[i],axis=2), np.expand_dims(data_logRelativePt[i],axis=2), np.expand_dims(data_isElectron[i],axis=2), np.expand_dims(data_isMuon[i],axis=2), np.expand_dims(data_isChargedHadron[i],axis=2), np.expand_dims(data_isNeutralHadron[i],axis=2), np.expand_dims(data_isPhoton[i],axis=2), np.expand_dims(data_charge[i],axis=2), np.expand_dims(data_logPt[i],axis=2), np.expand_dims(data_logE[i],axis=2)),axis=2) for i in range(0,len(fs))]
  #print("data", np.array(data).shape, np.array(data[0]).shape, np.array(data[0][0]).shape, np.array(data[0][0][0]).shape)
  """
  data = {}
  data['bes'] = np.concatenate([f['BES_vars'][:nevts] for f in fs])
  data["lab"] = np.concatenate([f['LabFrame_PFcands'][:nevts][...,[0,9,6,7,2,3,5,1]] for f in fs])
  data["b"] = np.concatenate([f['BottomFrame_PFcands'][:nevts][...,[0,9,6,7,2,3,5,1]] for f in fs])
  data["t"] = np.concatenate([f['TopFrame_PFcands'][:nevts][...,[0,9,6,7,2,3,5,1]] for f in fs])
  data["W"] = np.concatenate([f['WFrame_PFcands'][:nevts][...,[0,9,6,7,2,3,5,1]] for f in fs])
  data["Z"] = np.concatenate([f['ZFrame_PFcands'][:nevts][...,[0,9,6,7,2,3,5,1]] for f in fs])
  data["H"] = np.concatenate([f['HiggsFrame_PFcands'][:nevts][...,[0,9,6,7,2,3,5,1]] for f in fs])
  #print("Make label")
  label = np.concatenate([np.full(len(data[i]),i) for i in range(0,len(data))])

  #return (np.concatenate(data), np.concatenate(label))
  return (data,label)

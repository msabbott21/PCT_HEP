import os
import sys
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler, MinMaxScaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification


def shuffle_data(data, labels,global_pl=[],weights=[]):
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
  #return data[idx,:], labels[idx,:], idx
  if global_pl != []:
    return data[idx,:], labels[idx], global_pl[idx,:], idx
  elif weights == []:
    return data[idx,:], labels[idx],idx
  else:
    return data[idx,:], labels[idx], weights[idx],idx



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

def getNevents(h5_filename):
    filesize = h5py.File(h5_filename, 'r')["BES_vars"].shape[0]
    return filesize

def load_h5BEST(h5_filenames,num_points, begin, end):
# def load_h5BEST(h5_filenames,num_points, begin, end, debug=False):

   
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
    data_inv = [ f["PF_cands_AllFrame"][begin:end,0:num_points][...,[2,4,5,3,6,7]  ] for f in fs ]
    # ^This data structure is N_events x N_PFCands x [charge, isElectron, isMuon, isChargedHadron, isNeutralHadron, isPhoton]

    # Frame dependent shape: N_events x N_PFCands(50) x N_vars(11)
    # h5 Var indices: 0-deltaEta, 1-deltaPhi, 2-deltaR, 3-energy, 4-logEnergy, 5-logEnergyRatio, 6-logpT, 7-logpTRatio, 8-px, 9-py, 10-pz
    # data_lab = [ f["PF_cands_LabFrame"][:nevts,0:num_points][...,[0,1,2,5,7,6,4]] for f in fs ]
    data_lab = [ f["PF_cands_LabFrame"][begin:end,0:num_points][...,[0,1,2,5,7,6,4]] for f in fs ]
    # ^This data structure is N_events x N_PFCands x [deltaEta, deltaPhi, deltaR, logEnergyRatio, logpTRatio, logpT, logEnergy]

    data = [ np.concatenate( (data_lab[i], data_inv[i]), axis=2 ) for i in range(0,len(fs)) ]
    label = [np.full(len(data[i]),i) for i in range(0,len(data))]

    return (np.concatenate(data), np.concatenate(label))
    """
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
    """
    """
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

def load_lund(h5_filename):
  f = h5py.File(h5_filename,'r')
  data = f['data'][:]  
  label = f['truth_label'][:].astype(int)

  print("loaded {0} events".format(len(data)))
  return (data, label)


"""
Old h5 best implementation:

#print("load_h5BEST",num_points)
fs = [h5py.File(h5_filename,'r') for h5_filename in h5_filenames]
nevts=int(nevts)
nPFcands = num_points
labPFKey = "PF_cands_LabFrame"
#print("Make data") # The indices are 0-deltaEta, 9-deltaPhi, 6-pdgid, 7-charge, 2-px, 3-py, 5-pz, 1-e
#   data_knn = [f['LabFrame_PFcands'][:nevts,0:nPFcands][...,[0,9]] for f in fs] # deltaEta and deltaPhi
data_knn = [f['LabFrame_PFcands'][:nevts,0:nPFcands][...,[0,9]] for f in fs] # deltaEta and deltaPhi
#print("data_knn",np.array(data_knn).shape, np.array(data_knn[0]).shape, np.array(data_knn[0][0]).shape, np.array(data_knn[0][0][0]).shape)
#   data_deltaR = [np.sqrt(f['LabFrame_PFcands'][:nevts,0:nPFcands][...,0]**2 + f['LabFrame_PFcands'][:nevts,0:nPFcands][...,9]**2) for f in fs] # sqrt(deltaEta^2+deltaPhi^2)
data_deltaR = [np.sqrt(f['LabFrame_PFcands'][:nevts,0:nPFcands][...,0]**2 + f['LabFrame_PFcands'][:nevts,0:nPFcands][...,9]**2) for f in fs] # sqrt(deltaEta^2+deltaPhi^2)
#print("data_deltaR", np.array(data_deltaR).shape, np.array(data_deltaR[0]).shape, np.array(data_deltaR[0][0]).shape, np.array(data_deltaR[0][0][0]).shape)
data_logRelativeE = []
data_logRelativePt = []
data_logE = []
data_logPt = []
for f in fs:
    data_f_logRelativeE = np.zeros(np.array(f['LabFrame_PFcands'][:nevts,0:nPFcands][...,1]).shape)
    data_f_logRelativePt = np.zeros(data_f_logRelativeE.shape)
    data_f_logE = np.zeros(np.array(f['LabFrame_PFcands'][:nevts,0:nPFcands][...,1]).shape)
    data_f_logPt = np.zeros(data_f_logE.shape)
    for event in range(nevts):
        pfTempRelE = np.zeros(nPFcands)
        pfTempRelPt = np.zeros(nPFcands)
        pfTempE = np.zeros(nPFcands)
        pfTempPt = np.zeros(nPFcands)
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
# print("data_logRelativeE", np.array(data_logRelativeE).shape, np.array(data_logRelativeE[0]).shape, np.array(data_logRelativeE[0][0]).shape, np.array(data_logRelativeE[0][0][0]).shape)
# print("data_logRelativePt", np.array(data_logRelativePt).shape, np.array(data_logRelativePt[0]).shape, np.array(data_logRelativePt[0][0]).shape, np.array(data_logRelativePt[0][0][0]).shape)
# print("data_logE", np.array(data_logE).shape, np.array(data_logE[0]).shape, np.array(data_logE[0][0]).shape, np.array(data_logE[0][0][0]).shape)
# print("data_logPt", np.array(data_logPt).shape, np.array(data_logPt[0]).shape, np.array(data_logPt[0][0]).shape, np.array(data_logPt[0][0][0]).shape)
# data_logRelativeE = [np.log((np.divide(f['LabFrame_PFcands'][:nevts][...,1], np.expand_dims(f['BES_vars'][:nevts,23],axis=1)))) for f in fs] # jetAK8_e is index 23
# data_logRelativePt = [np.log((np.sqrt((f['LabFrame_PFcands'][:nevts][...,2]**2 + f['LabFrame_PFcands'][:nevts][...,3]**2))/f['BES_vars'][:nevts][...,53])) for f in fs] # jetAK8_pt is index 53
# data_logMomenta = [numpy.log(f['LabFrame_PFcands'][:nevts][...,[2,3,5]]) for f in fs] # px, py, pz
# data_logPt = [np.log(np.sqrt((f['LabFrame_PFcands'][:nevts][...,2]**2 + f['LabFrame_PFcands'][:nevts][...,3]**2))) for f in fs] # px, py
# data_logE = [np.log(f['LabFrame_PFcands'][:nevts][...,1]) for f in fs] # e

data_isElectron = [abs(f['LabFrame_PFcands'][:nevts,0:nPFcands][...,6])==11 for f in fs]
#print("data_isElectron", np.array(data_isElectron).shape, np.array(data_isElectron[0]).shape, np.array(data_isElectron[0][0]).shape, np.array(data_isElectron[0][0][0]).shape)
data_isMuon = [abs(f['LabFrame_PFcands'][:nevts,0:nPFcands][...,6])==13 for f in fs]
#print("data_isMuon", np.array(data_isMuon).shape, np.array(data_isMuon[0]).shape, np.array(data_isMuon[0][0]).shape, np.array(data_isMuon[0][0][0]).shape)
data_isChargedHadron = [(abs(f['LabFrame_PFcands'][:nevts,0:nPFcands][...,6])==211) | (abs(f['LabFrame_PFcands'][:nevts,0:nPFcands][...,6])==321) for f in fs]
#print("data_isChargedHadron", np.array(data_isChargedHadron).shape, np.array(data_isChargedHadron[0]).shape, np.array(data_isChargedHadron[0][0]).shape, np.array(data_isChargedHadron[0][0][0]).shape)
data_isNeutralHadron = [(abs(f['LabFrame_PFcands'][:nevts,0:nPFcands][...,6])==111) | (abs(f['LabFrame_PFcands'][:nevts,0:nPFcands][...,6])==130) | (abs(f['LabFrame_PFcands'][:nevts,0:nPFcands][...,6])==310) | (abs(f['LabFrame_PFcands'][:nevts,0:nPFcands][...,6])==311) for f in fs]
#print("data_isNeutralHadron", np.array(data_isNeutralHadron).shape, np.array(data_isNeutralHadron[0]).shape, np.array(data_isNeutralHadron[0][0]).shape, np.array(data_isNeutralHadron[0][0][0]).shape)
data_isPhoton = [abs(f['LabFrame_PFcands'][:nevts,0:nPFcands][...,6])==22 for f in fs]
#print("data_isPhoton", np.array(data_isPhoton).shape, np.array(data_isPhoton[0]).shape, np.array(data_isPhoton[0][0]).shape, np.array(data_isPhoton[0][0][0]).shape)
data_charge = [f['LabFrame_PFcands'][:nevts,0:nPFcands][...,7] for f in fs]
#print("data_charge", np.array(data_charge).shape, np.array(data_charge[0]).shape, np.array(data_charge[0][0]).shape, np.array(data_charge[0][0][0]).shape)
data = [np.concatenate((data_knn[i], np.expand_dims(data_deltaR[i],axis=2), np.expand_dims(data_logRelativeE[i],axis=2), np.expand_dims(data_logRelativePt[i],axis=2), np.expand_dims(data_isElectron[i],axis=2), np.expand_dims(data_isMuon[i],axis=2), np.expand_dims(data_isChargedHadron[i],axis=2), np.expand_dims(data_isNeutralHadron[i],axis=2), np.expand_dims(data_isPhoton[i],axis=2), np.expand_dims(data_charge[i],axis=2), np.expand_dims(data_logPt[i],axis=2), np.expand_dims(data_logE[i],axis=2)),axis=2) for i in range(0,len(fs))]
#print("data", np.array(data).shape, np.array(data[0]).shape, np.array(data[0][0]).shape, np.array(data[0][0][0]).shape)
#data = [f['LabFrame_PFcands'][:nevts][...,[0,9,6,7,2,3,5,1]] for f in fs]
#print("Make label")
label = [np.full(len(data[i]),i) for i in range(0,len(data))]

return (np.concatenate(data), np.concatenate(label))
"""
import os
import pickle
import sys
import random
import gzip
import glob
import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp
import scipy
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os
from PIL import Image


pathImageNet = "../../datasets/ImageNet/ILSVRC/Data/CLS-LOC/train/"
# pathImageNet = "../../datasets/ImageNet/ILSVRC/Data/CLS-LOC/val/"


all_images = glob.glob(pathImageNet + "**/*.JPEG", recursive=True)

# load precomputed log density over a 1024x1024 image

#Log density predictions
result = {}

ct = 0
centerbias_template = np.load('centerbias.npy')  
with tf.Session() as sess:
    check_point = '../deep_gaze/DeepGazeII.ckpt'  # DeepGaze II
    #check_point = 'ICF.ckpt'  # ICF
    new_saver = tf.train.import_meta_graph('{}.meta'.format(check_point))
    new_saver.restore(sess, check_point)
    for image in all_images:
        #skip if we've already computed
        if os.path.isfile(image.split(".JPEG")[0]+".saliency.npy.gz"):
            continue
        
        img = plt.imread(image)
        if len(img.shape)==2:
            img = np.stack([img,img,img],axis=2)
        # rescale to match image size
        centerbias = zoom(centerbias_template, (img.shape[0]/1024., img.shape[1]/1024.), order=0, mode='nearest')
        # renormalize log density
        centerbias -= logsumexp(centerbias)
        image_data = np.array([img])
        centerbias_data = np.repeat([centerbias],len(image_data),axis=0)[:, :, :, np.newaxis]  # BHWC, 1 channel (log density)
        #tf.reset_default_graph()
        log_density = tf.get_collection('log_density')[0]
        centerbias_tensor = tf.get_collection('centerbias_tensor')[0]
        input_tensor = tf.get_collection('input_tensor')[0]
        try:
            log_density_prediction = sess.run(log_density, {
                input_tensor: image_data,
                centerbias_tensor: centerbias_data,
            })
            ldp = log_density_prediction[0,:,:,0]
            with gzip.open(image.split(".JPEG")[0]+".saliency.npy.gz", "wb") as f:
                np.save(file=f, arr=ldp)
        except Exception as e:
            print("ERROR at: " + image)
            print(e)
            continue
#         key = "/".join(image.split("/")[7:])
#         value = np.unravel_index(np.argmax(ldp[0, :, :, 0]),ldp[0, :, :, 0].shape)
#         result[key] = ldp
#         ct+=1
#         if ct%1000==0:
#             pickle_out = open("predictions_pt"+str(ct//1000)+".pkl","wb") 
#             pickle.dump(result, pickle_out) 
#             pickle_out.close()
#             result = {}



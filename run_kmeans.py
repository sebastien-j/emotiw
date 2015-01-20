#!/usr/bin/env python

from pylab import *

import numpy as np
import random as rnd
import os
from os import chdir
import scipy
from scipy import cluster

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-training-patches", type=int, default=10)
    parser.add_argument("--patchsize", type=int, default=8)
    parser.add_argument("--var-threshold", type=float, default=0.9)
    parser.add_argument("--num-centroids", type=int, default=400)
    parser.add_argument("--v-min", type=int, default=44)
    parser.add_argument("--v-max", type=int, default=82)
    parser.add_argument("--h-min", type=int, default=16)
    parser.add_argument("--h-max", type=int, default=66)
    parser.add_argument("--v-sections", type=int, default=4)
    parser.add_argument("--h-sections", type=int, default=4)
    parser.add_argument("--root-dir", type=str, default='/data/lisa/exp/ebrahims/emotiw_pipeline/EmotiW2014_Train/facetubes_96x96')
    parser.add_argument("--save-centroids", type=str, default='/data/lisatmp3/jeasebas/emotiw/centroids.npy')
    parser.add_argument("--save-mean-inter", type=str, default='/data/lisatmp3/jeasebas/emotiw/mean_inter.npy')
    parser.add_argument("--save-mean-inter", type=str, default='/data/lisatmp3/jeasebas/emotiw/V_list.npy')
    
    return parser.parse_args()

def main():
    args = parse_args()

    num_training_patches = args.num_training_patches
    patchsize = args.patchsize
    eps = 0.000000001
    var_treshold = args.var_threshold
    size_ = patchsize*patchsize*3
    num_centroids = args.num_centroids
    v_min = args.v_min
    v_max = args.v_max
    h_min = args.h_min
    h_max = args.h_max
    v_sections = args.v_sections
    h_sections = args.h_sections

    v_size = v_max - v_min
    h_size = h_max - h_min
    
    root_dir = args.root_dir

    training = []
    for dir in sorted(os.listdir(root_dir)):
        for file in sorted(os.listdir(os.path.join(root_dir, dir))):
            training.append(imread(os.path.join(root_dir, dir, file))[v_min:v_max, h_min:h_max])

    training = asarray(training)
    num_training_images = shape(training)[0]

    centroids = []
    V_list = []
    training_patches_inter = zeros((v_sections*h_sections,num_training_images*num_training_patches,size_))

    for i_v in xrange(v_sections):
        for i_h in xrange(h_sections):
            print i_v, i_h
            training_patches = zeros((num_training_images*num_training_patches,size_))
            
            index = 0
            for i in range(num_training_images*num_training_patches):
                num0 = i_v*((v_size-patchsize)/v_sections)+randint((v_size-patchsize)/v_sections)
                num1 = i_h*((h_size-patchsize)/h_sections)+randint((h_size-patchsize)/h_sections)
                patch = training[i/num_training_patches,num0:num0+patchsize,num1:num1+patchsize,:].copy()
                patch -= mean(patch) # CHECK AXES (MAY BE WRONG)
                patch /= (std(patch)+eps)
                patch = reshape(patch, (1,size_)) 
                training_patches[i]=patch.copy()        
            
            training_patches_inter[i_v*h_sections+i_h] = training_patches.copy()
            
            for j in range (size_):
                training_patches[:,j] -= mean(training_patches[:,j])

            cov = dot(training_patches.T,training_patches)/(num_training_images*num_training_patches)

            svd_ = svd(cov)

            eigval = svd_[1]
            U = svd_[0]

            cutoff = var_treshold*sum(eigval)

            total = 0.
            dim = 0

            while total < cutoff:
                total += eigval[dim]
                dim += 1

            eigval_ = eigval[0:dim]

            U_ = U[:,0:dim]

            V_ = (U_.T).copy()

            for k in range(dim):
                V_[k] /= (sqrt(eigval_[k])+eps)
                
            V_list.append(V_.copy())

            white_training_patches=dot(V_,training_patches.T) #each white patch is a column
            
            while True:
                try:
                    temp_clusters =scipy.cluster.vq.kmeans2(white_training_patches.T,num_centroids,iter=25,minit='points', missing='raise')
                    break
                except:
                    print "ClusterError. Doing kmeans again."
            centroids.append(temp_clusters[0].T)

    mean_inter = zeros((v_sections*h_sections,size_))
    for i_v in xrange(v_sections):
        for i_h in xrange(h_sections):
            mean_inter[i_v*h_sections+i_h] = mean(training_patches_inter[i_v*h_sections+i_h],0)
    
    np.save(args.save_centroids, centroids)
    np.save(args.save_mean_inter, mean_inter)
    np.save(args.save_v_list, V_list)

if __name__ == "__main__":
    main()
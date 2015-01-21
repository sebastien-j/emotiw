import os
from pylab import *
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batchsize", type=int, default=50)
    parser.add_argument("--patchsize", type=int, default=8)
    parser.add_argument("--num-centroids", type=int, default=400)
    parser.add_argument("--v-min", type=int, default=44)
    parser.add_argument("--v-max", type=int, default=82)
    parser.add_argument("--h-min", type=int, default=16)
    parser.add_argument("--h-max", type=int, default=66)
    parser.add_argument("--v-sections", type=int, default=4)
    parser.add_argument("--h-sections", type=int, default=4)
    parser.add_argument("--root-dir", type=str, default='/data/lisa/exp/ebrahims/emotiw_pipeline/EmotiW2014_train/facetubes_96x96')
    parser.add_argument("--centroids", type=str, default='/data/lisatmp3/jeasebas/emotiw/centroids.npy')
    parser.add_argument("--mean-inter", type=str, default='/data/lisatmp3/jeasebas/emotiw/mean_inter.npy')
    parser.add_argument("--v-list", type=str, default='/data/lisatmp3/jeasebas/emotiw/V_list.npy')
    parser.add_argument("--save-features", type=str)   

def main():
    args = parse_args()

    patchsize = args.patchsize
    eps = 0.000000001
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

    mean_inter = np.load(args.mean_inter)
    centroids = np.load(args.centroids)
    V_list = np.load(args.v_list)

    def feature_extract(array_im): #shape num_im, v_size, h_size, 3
        
        cur_patches = zeros((array_im.shape[0],v_sections*h_sections,((v_size-patchsize)/v_sections)*((h_size-patchsize)/h_sections),size_))
        cur_white_patches = []
        
        for i_v in xrange(v_sections):
            for i_h in xrange(h_sections):
                for i in xrange((v_size-patchsize)/v_sections):
                    for j in xrange((h_size-patchsize)/h_sections):
                        patch = array_im[:,i_v*(v_size-patchsize)/v_sections+i:i_v*(v_size-patchsize)/v_sections+i+patchsize,i_h*(h_size-patchsize)/h_sections+j:i_h*(h_size-patchsize)/h_sections+j+patchsize].copy()
                        #print shape(patch)
                        #print (array_im.shape[0],size_)                    
                        patch = reshape(patch, (array_im.shape[0],size_)).transpose(1,0)
                        patch -= mean(patch,0)
                        patch /= (std(patch,0)+eps)
                        patch = patch.transpose(1,0)
                        cur_patches[:,i_v*h_sections+i_h,(h_size-patchsize)/h_sections*i+j] = patch.copy()

        for i_v in xrange(v_sections):
            for i_h in xrange(h_sections):
                cur_patches[:,i_v*h_sections+i_h] -= mean_inter[i_v*h_sections+i_h]
        
        cur_patches = cur_patches.transpose(0,1,3,2)

        for i_v in xrange(v_sections):
            for i_h in xrange(h_sections):
                cur_white_patches.append((dot(V_list[i_v*h_sections+i_h],cur_patches[:,i_v*h_sections+i_h])).transpose(1,0,2))

        feature_map = zeros((array_im.shape[0],v_sections*h_sections,((v_size-patchsize)/v_sections)*((h_size-patchsize)/h_sections),num_centroids))
        
        for i_v in xrange(v_sections):
            for i_h in xrange(h_sections):
                for j in xrange(((v_size-patchsize)/v_sections)*((h_size-patchsize)/h_sections)):
                    z = zeros((array_im.shape[0],num_centroids))
                    f = zeros((array_im.shape[0],num_centroids))
                    for k in range(num_centroids):
                        z[:,k] = sqrt(sum((cur_white_patches[i_v*h_sections+i_h][:,:,j]-centroids[i_v*h_sections+i_h][:,k])**2,axis=1))
                    mu = mean(z,1) # one mean per image
                    for k in range(num_centroids):
                        f[:,k] = clip(mu - z[:,k],0,inf)
                    feature_map[:,i_v*h_sections+i_h,j,:] = f.copy()   
        
        pooled_features = zeros((array_im.shape[0],v_sections*h_sections*num_centroids))

        for i_v in xrange(v_sections):
            for i_h in xrange(h_sections):
                pooled_features[:,(i_v*h_sections+i_h)*num_centroids:(i_v*h_sections+i_h+1)*num_centroids] = mean(feature_map[:,i_v*h_sections+i_h],1)
        
        if array_im.shape[0] == 1:
            pooled_features = reshape(pooled_features, (v_sections*h_sections*num_centroids))
        
        return pooled_features.T

    data = []

    for dir in sorted(args.root_dir):
        for file in sorted(os.listdir(os.path.join(root_dir,dir))):
            data.append(imread(os.path.join(root_dir,dir,file))[v_min:v_max,h_min:h_max])

    data = asarray(data)
    num_images = shape(data)[0]

    features = zeros((v_sections*h_sections*num_centroids,num_images))

    batchsize = args.batchsize

    for j in xrange(num_images/batchsize):
        print j
        features[:,batchsize*j:batchsize*(j+1)] = feature_extract(data[batchsize*j:batchsize*(j+1)])

    if num_test_images%batchsize > 1:
        features[:,num_images - num_images%batchsize:num_images] = feature_extract(data[num_images - num_images%batchsize:num_images])
    elif num_images%batchsize == 1:
        features[:,num_images - num_images%batchsize:num_test_images] = feature_extract(data[num_images - num_images%batchsize:num_images])[:,None]
    
    if args.save_features:
        np.save(args.save_features, features)


if __name__ == "__main__":
    main()
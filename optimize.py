import os
from pylab import *
import numpy as np
import argparse
import logreg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-centroids", type=int, default=400)
    parser.add_argument("--v-sections", type=int, default=4)
    parser.add_argument("--h-sections", type=int, default=4)
    parser.add_argument("--v-min", type=int, default=55)
    parser.add_argument("--v-max", type=int, default=85)
    parser.add_argument("--h-min", type=int, default=30)
    parser.add_argument("--h-max", type=int, default=70)
    parser.add_argument("--root-dir", type=str, default='/data/lisa/exp/ebrahims/emotiw_pipeline/EmotiW2014_train/facetubes_96x96')
    parser.add_argument("--label-dir", type=str, default='/data/lisa/exp/ebrahims/emotiw_pipeline/EmotiW2014_train/Labels')
    parser.add_argument("--save-weights", type=str, default='/data/lisatmp3/jeasebas/emotiw/weights.npy')
    parser.add_argument("--save-biases", type=str, default='/data/lisatmp3/jeasebas/emotiw/biases.npy')
    parser.add_argument("--features", type=str, default='/data/lisatmp3/jeasebas/emotiw/train_features.npy')
    parser.add_argument("--wc", type=float, default=1e-3)
    parser.add_argument("--sgd", action="store_true")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--bs", type=int, default=50)
    parser.add_argument("--dropout", action="store_true")
    parser.add_argument("--dropout-fixed", type=int, default=10)
    parser.add_argument("--maxnumlinesearch", type=int, default=500)

    return parser.parse_args()

def main():
    args = parse_args()

    num_centroids = args.num_centroids
    v_sections = args.v_sections
    h_sections = args.h_sections
    v_min = args.v_min
    v_max = args.v_max
    h_min = args.h_min
    h_max = args.h_max
    
    label_dict = {'Angry':0, 'Disgust':1, 'Fear':2, 'Happy':3, 'Sad':4, 'Surprise':5, 'Neutral':6}
    data = []
    labels = []

    root_dir = args.root_dir
    label_dir = args.label_dir
    for dir in sorted(os.listdir(root_dir)):
        with open(os.path.join(label_dir, dir)) as f:
            cur_label = zeros(7)
            cur_label[label_dict[f.readline().strip()]] = 1
        for file in sorted(os.listdir(os.path.join(root_dir,dir))):
            data.append(imread(os.path.join(root_dir,dir,file))[v_min:v_max,h_min:h_max])
            labels.append(cur_label)
    data = asarray(data)
    labels = asarray(labels).T
    num_images = shape(data)[0]

    order = arange(num_images)
    shuffle(order)

    features = np.load(args.features)
    assert num_images == np.shape(features)[1]
    features_shuffled = zeros((v_sections*h_sections*num_centroids,num_images))

    if not args.sgd:
        for j in xrange(num_images):
            features_shuffled[:,j] = features[:,order[j]]

        labels_shuffled = zeros((7,num_images))

        for j in xrange(num_images):
            labels_shuffled[:,j] = labels[:,order[j]]

    numclasses = 7
    wc = args.wc

    lr = logreg.Logreg(numclasses, features_shuffled.shape[0])
    if not args.sgd:
        if args.dropout:
            original_features_shuffled = features_shuffled.copy()
            features_shape = shape(features_shuffled)
            for i in xrange(args.maxnumlinesearch/args.dropout_fixed):
                print i
                mask = np.random.randint(2, size = features_shape)
                features_shuffled = original_features_shuffled * mask
                lr.train_cg(features_shuffled,labels_shuffled,verbose=True,weightcost=wc,maxnumlinesearch=args.dropout_fixed)
        else:
            lr.train_cg(features_shuffled,labels_shuffled,verbose=True,weightcost=wc,maxnumlinesearch=args.maxnumlinesearch)
    else:
        lr.train_minibatch(features, labels, wc, args.epochs, args.bs, args.lr, dropout=args.dropout)

    np.save(args.save_weights, lr.weights)
    np.save(args.save_biases, lr.biases)

if __name__ == "__main__":
    main()
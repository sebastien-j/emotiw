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
    parser.add_argument("--root-dir", type=str, default='/data/lisa/exp/ebrahims/emotiw_pipeline/EmotiW2014_train/facetubes_96x96')
    parser.add_argument("--label-dir", type=str, default='/data/lisa/exp/ebrahims/emotiw_pipeline/EmotiW2014_train/Labels')
    parser.add_argument("--weights", type=str, default='/data/lisatmp3/jeasebas/emotiw/weights.npy', nargs='+')
    parser.add_argument("--biases", type=str, default='/data/lisatmp3/jeasebas/emotiw/biases.npy', nargs='+')
    parser.add_argument("--features", type=str, default='/data/lisatmp3/jeasebas/emotiw/train_features.npy')
    parser.add_argument("--save-probabilities", type=str, default='/data/lisatmp3/jeasebas/emotiw/train_probabilities.npy')
    parser.add_argument("--dropout", action="store_true")

    return parser.parse_args()

def main():
    args = parse_args()

    num_centroids = args.num_centroids
    v_sections = args.v_sections
    h_sections = args.h_sections
    
    label_dict = {'Angry':0, 'Disgust':1, 'Fear':2, 'Happy':3, 'Sad':4, 'Surprise':5, 'Neutral':6}
    numclasses = len(label_dict)

    lrs = []
    assert len(args.weights) == len(args.biases)
    for i in xrange(len(args.weights)):
        lrs.append(logreg.Logreg(numclasses, v_sections*h_sections*num_centroids))
        lrs[i].weights = np.load(args.weights[i])
        lrs[i].biases = np.load(args.biases[i])

    features = np.load(args.features)

    root_dir = args.root_dir

    probabilities = []
    start = 0
    for dir in sorted(os.listdir(root_dir)):
        end = start + len(sorted(os.listdir(os.path.join(root_dir,dir))))
        if end > start:
            tmp_probs = 0.0
            for i in xrange(len(args.weights)):
                tmp_probs += mean(lrs[i].probabilities(features[:,start:end], multiplier=float(args.dropout)),1)
            probabilities.append(tmp_probs)
        else:
            probabilities.append(asarray(7*[1./7]))
        start = end
    probabilities = asarray(probabilities)
    
    np.save(args.save_probabilities, probabilities)

    if args.label_dir:
        correct = 0
        total = 0
        label_dir = args.label_dir
        for i, dir in enumerate(sorted(os.listdir(root_dir))):
            with open(os.path.join(label_dir, dir)) as f:
                if label_dict[f.readline().strip()] == np.argmax(probabilities[i]):
                    correct += 1
        total = i + 1
        print correct, total, (correct * 100. / total)

if __name__ == "__main__":
    main()
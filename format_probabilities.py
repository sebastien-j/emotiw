import os
from pylab import *
import numpy as np
import argparse
import logreg

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, default='/data/lisatmp3/jeasebas/emotiw/EmotiW2014_train/facetubes_96x96')
    parser.add_argument("--probabilities", type=str, default='/data/lisatmp3/jeasebas/emotiw/train_probabilities.npy')
    parser.add_argument("--save-file", type=str)

    return parser.parse_args()

def main():
    args = parse_args()
    
    label_dict = {'Angry':0, 'Disgust':1, 'Fear':2, 'Happy':3, 'Sad':4, 'Surprise':5, 'Neutral':6}
    invert_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'} 
    numclasses = len(label_dict)

    with open(args.save_file, 'w') as f:
        for i, dir in enumerate(sorted(os.listdir(root_dir))):
            cur_str = dir + '.full.pca.pkl' + ' ' + invert_dict(np.argmax(args.probabilities[i]))
            for j in xrange(numclasses):
                cur_str += (' ' + str(args.probabilities[i,j]))
            cur_str += ('\n')
            f.write(cur_str)            

if __name__ == "__main__":
    main()
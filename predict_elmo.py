from numpy.random import seed
seed(3)
#from tensorflow import set_random_seed
#set_random_seed(3)
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input, Bidirectional, Lambda, TimeDistributed, Masking, Average
#from keras_contrib.layers import CRF
from keras import optimizers
import keras.backend as K
import csv
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix, f1_score
from allennlp.commands.elmo import ElmoEmbedder
from math import ceil
from extra.apply_vecmap_transform import vecmap
from nerutils import load_data, embed_elmo, pad_labels



def generate_batch_data(inputfile, batch_size, args):
    elmo = ElmoEmbedder(args.options, args.weights, -1)
    if args.mat0:
        W0 = {}
        W1 = {}
        W2 = {}
        mapmat = np.load(args.mat0)
        W0['src'] = mapmat['wx2']
        W0['trg'] = mapmat['wz2']
        W0['s'] = mapmat['s']
        mapmat = np.load(args.mat1)
        W1['src'] = mapmat['wx2']
        W1['trg'] = mapmat['wz2']
        W1['s'] = mapmat['s']
        mapmat = np.load(args.mat2)
        W2['src'] = mapmat['wx2']
        W2['trg'] = mapmat['wz2']
        W2['s'] = mapmat['s']
        mapmat = None
        xlingual = [W0, W1, W2]
    else:
        xlingual = [False,]*3
    while True: # it needs to be infinitely iterable            
        x,y = load_data(inputfile)
        print("INPUT SIZES X AND Y", len(x), len(y))
        assert len(x) == len(y)
        newxval = []
        yval = []
        for i in range(len(y)):
            newxval.append(x[i])
            yval.append(y[i])
            assert len(newxval) == len(yval)
            if i > 0 and i % batch_size == 0:
                xval0, xval1, xval2 = embed_elmo(newxval, elmo, xlingual, lang=args.evlang)
                ypadded = pad_labels(yval)
                yield ([np.array(xval0), np.array(xval1), np.array(xval2)], np.array(ypadded))
                newxval = []
                yval = []
        if len(newxval) > 0:
            xval0, xval1, xval2 = embed_elmo(newxval, elmo, xlingual, lang=args.evlang)
            ypadded = pad_labels(yval)
            yield ([np.array(xval0), np.array(xval1), np.array(xval2)], np.array(ypadded))

    
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--test_file", default=None, type=str, required=True)
    parser.add_argument("--options", default=None, type=str, required=True, help="elmo options file")
    parser.add_argument("--weights", default=None, type=str, required=True, help="elmo weights file")
    #parser.add_argument("--train_len", default=0, type=int, required=True)
    parser.add_argument("--test_len", default=0, type=int, required=True)
    parser.add_argument('--mat0', help='mapping matrices for layer0 (.npz), optional')
    parser.add_argument('--mat1', help='mapping matrices for layer1 (.npz), optional')
    parser.add_argument('--mat2', help='mapping matrices for layer2 (.npz), optional')
    #parser.add_argument('--trlang', default='trg', type=str, help='src or trg when mapping train file language')
    parser.add_argument('--evlang', default='src', type=str, help='src or trg when mapping test file language')    
    parser.add_argument("--bs", default=64, type=int, help="batch size")
    parser.add_argument("--save", default="elmo_new_ner_model", type=str, help="path to trained elmo NER model")
    args = parser.parse_args()
    


    max_len = None
    # NN
    model = load_model(args.save)
    y_predict = model.predict_generator(generate_batch_data(args.test_file, args.bs, args), steps=ceil(args.test_len/args.bs))

    _, y_ev = load_data(args.test_file)

    y_ev_i = []
    y_pr_i = []
    for s in range(len(y_ev)):
        for w in range(len(y_ev[s])):
            y_ev_i.append(np.argmax(y_ev[s][w]))
            y_pr_i.append(np.argmax(y_predict[s][w]))

    
    print('---***---')
    print(confusion_matrix(y_ev_i, y_pr_i))
    print(f1_score(y_ev_i, y_pr_i, average='micro'))
    print(f1_score(y_ev_i, y_pr_i, average='macro'))
    print(f1_score(y_ev_i, y_pr_i, average=None))


if __name__ == "__main__":
    main()    

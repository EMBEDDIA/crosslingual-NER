from numpy.random import seed
seed(3)
#from tensorflow import set_random_seed
#set_random_seed(3)
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input, Bidirectional, Lambda, TimeDistributed, Masking, Average
#from keras_contrib.layers import CRF
from keras import optimizers, losses
import keras.backend as K
import csv
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix, f1_score
from allennlp.commands.elmo import ElmoEmbedder
from math import ceil
import sys
from nerutils import embed_elmogan, load_data, pad_labels

def htanh(a):
    return K.maximum(-1.0, K.minimum(1.0, a))

def csd2(x,y):
    return 1.0*losses.cosine_proximity(x,y)+0.5*losses.mean_absolute_error(x,y)


def generate_batch_data(inputfile, batch_size, args):
    elmo = ElmoEmbedder(args.options, args.weights, -1)
    if args.mat0:
        sys.stderr.write("loading mapping models\n")
        my_funcs = {'htanh':htanh, 'csd2':csd2, 'cosine_proximity':losses.cosine_proximity}
        with tf.device("cpu:0"):
            W0 = load_model(args.mat0, custom_objects=my_funcs)
            W1 = load_model(args.mat1, custom_objects=my_funcs)
            W2 = load_model(args.mat2, custom_objects=my_funcs)
        xlingual = [W0, W1, W2]
    else:
        xlingual = [False,]*3
    while True: # it needs to be infinitely iterable            
        x,y = load_data(inputfile)
        sys.stderr.write("loading eval data\n")
        print("INPUT SIZES X AND Y", len(x), len(y))
        assert len(x) == len(y)
        newxval = []
        yval = []
        for i in range(len(y)):
            newxval.append(x[i])
            yval.append(y[i])
            assert len(newxval) == len(yval)
            if i > 0 and i % batch_size == 0:
                xval0, xval1, xval2 = embed_elmogan(newxval, elmo, xlingual, args)
                ypadded = pad_labels(yval)
                yield ([np.array(xval0), np.array(xval1), np.array(xval2)], np.array(ypadded))
                newxval = []
                yval = []
        if len(newxval) > 0:
            xval0, xval1, xval2 = embed_elmogan(newxval, elmo, xlingual, args)
            ypadded = pad_labels(yval)
            yield ([np.array(xval0), np.array(xval1), np.array(xval2)], np.array(ypadded))

    
def main():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--eval_file", default=None, type=str, required=True)
    parser.add_argument("--options", default=None, type=str, required=True)
    parser.add_argument("--weights", default=None, type=str, required=True)
    parser.add_argument("--eval_len", default=0, type=int, required=True)
    parser.add_argument('--mat0', help='maping NN model for layer0 (.h5)')
    parser.add_argument('--mat1', help='mapping NN model for layer1 (.h5)')
    parser.add_argument('--mat2', help='mapping NN model for layer2 (.h5)')
    parser.add_argument("--bs", default=64, type=int, help="batch size")
    parser.add_argument("--save", default="elmo_new_ner_model", type=str)
    parser.add_argument("--direction", type=int, required=True, choices=[0,1], help='Given a model xx-yy to yy-xx choose 0 for xx->yy or 1 for yy->xx')
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()
    


    max_len = None
    # NN
    sys.stderr.write("Loading NER model\n")
    with tf.device("gpu:0"):
        model = load_model(args.save)
    sys.stderr.write("beginning predict process\n")
    y_predict = model.predict_generator(generate_batch_data(args.eval_file,args.bs,args), steps=ceil(args.eval_len/args.bs))
    _, y_ev = load_data(args.eval_file)

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

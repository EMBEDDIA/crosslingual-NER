from numpy.random import seed
seed(3)
#from tensorflow import set_random_seed
#set_random_seed(3)
import tensorflow as tf
from keras.models import Model
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
from nerutils import load_data, pad_labels, apply_vecmap, normalize


def embed_elmo(sentences, elmo_embedder, xlingual, normal=False, lang=''):
    emb_batch = 256 #128 for et, 2<=n<8 for sv, en=?, others can use higher probably
    #swedish has problem around sentences 1500-1800 in train (extra high ram usage)
    #embedded = map(elmo_embedder.embed_sentence, sentences)
    apply_mapping = apply_vecmap

    max_seqlen = max(len(s) for s in sentences) if sentences else 0
    if max_seqlen == 0:
        return []
    emb0 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
    emb1 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
    emb2 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
    emba = []
    emb = elmo_embedder.embed_batch(sentences)
    for x,sentence in enumerate(emb):
        if normal:
            for i in range(3):
                normalize(sentence[i])
        seqlen = sentence[0].shape[0]
        emb0[x, 0:seqlen, :] = apply_mapping(sentence[0], xlingual[0], lang)
        emb1[x, 0:seqlen, :] = apply_mapping(sentence[1], xlingual[1], lang)
        emb2[x, 0:seqlen, :] = apply_mapping(sentence[2], xlingual[2], lang)
    embedded = [emb0, emb1, emb2]
    return embedded

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", default=None, type=str, required=True)
    parser.add_argument("--eval_file", default=None, type=str, required=True)
    parser.add_argument("--options", default=None, type=str, required=True, help="elmo options file")
    parser.add_argument("--weights", default=None, type=str, required=True, help="elmo weights file")
    parser.add_argument("--train_len", default=0, type=int, required=True, help="number of tokens in train file")
    parser.add_argument("--eval_len", default=0, type=int, required=True, help="number of tokens in evaluation file")
    parser.add_argument('--mat0', help='mapping matrices for layer0 (.npz), do not specify for monolingual setting')
    parser.add_argument('--mat1', help='mapping matrices for layer1 (.npz), do not specify for monolingual')
    parser.add_argument('--mat2', help='mapping matrices for layer2 (.npz), do not specify for monolingual')
    parser.add_argument('--trlang', default='trg', type=str, help='src or trg when mapping train file language')
    parser.add_argument('--evlang', default='src', type=str, help='src or trg when mapping eval file language')    
    parser.add_argument("--bs", default=64, type=int, help="batch size")
    parser.add_argument("--epoch", default=3, type=int, help="number of epochs to train for")
    parser.add_argument("--save", default="elmo_new_ner_model", type=str, help="Filename to save the NER model to")
    args = parser.parse_args()
    

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
                    xval0, xval1, xval2 = embed_elmo(newxval, elmo, xlingual, lang=args.trlang)
                    ypadded = pad_labels(yval)
                    yield ([np.array(xval0), np.array(xval1), np.array(xval2)], np.array(ypadded))
                    newxval = []
                    yval = []
            if len(newxval) > 0:
                xval0, xval1, xval2 = embed_elmo(newxval, elmo, xlingual, lang=args.trlang)
                ypadded = pad_labels(yval)
                yield ([np.array(xval0), np.array(xval1), np.array(xval2)], np.array(ypadded))

    max_len = None
    
    # NN
    input_cnn = Input(shape=(max_len,1024), dtype="float32")
    mask_cnn = Masking(mask_value=-999., input_shape=(max_len, 1024)) (input_cnn)
    input_lstm1 = Input(shape=(max_len,1024), dtype="float32")
    mask_lstm1 = Masking(mask_value=-999., input_shape=(max_len, 1024)) (input_lstm1)
    input_lstm2 = Input(shape=(max_len,1024), dtype="float32")
    mask_lstm2 = Masking(mask_value=-999., input_shape=(max_len, 1024)) (input_lstm2)
    input_layer = Input(shape=(max_len,1024), dtype="float32")
    avglayer = Average()([mask_cnn, mask_lstm1, mask_lstm2])
    bilstm1 = Bidirectional(LSTM(units=2048, return_sequences=True)) (avglayer)
    bilstm2 = Bidirectional(LSTM(units=2048, return_sequences=True)) (bilstm1)
    out = TimeDistributed(Dense(4, activation="softmax")) (bilstm2)
    
    adam = optimizers.Adam(lr=1e-4, decay=1e-5)
    model = Model(inputs=[input_cnn, input_lstm1, input_lstm2], outputs=out)
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
    
    model.fit_generator(generate_batch_data(args.train_file, args.bs, args), steps_per_epoch=ceil(args.train_len/args.bs), epochs=args.epoch, validation_data=generate_batch_data(args.eval_file, args.bs, args), validation_steps=ceil(args.eval_len/args.bs) )

    model.save(args.save)

if __name__ == "__main__":
    main()    

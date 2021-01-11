from numpy.random import seed
seed(3)
from tensorflow import set_random_seed
set_random_seed(3)
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Input, Bidirectional, Lambda, TimeDistributed, Masking
#from keras_contrib.layers import CRF
from keras import optimizers
import keras.backend as K
import csv
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix, f1_score
from math import ceil
import sys
# LOAD DATA
## read tsv to python lists
## where x has tokenized sentences
## and y has 1-hot encoded categories
### 1-hot encoding preferably a separate function

# CONSTRUCT NN
## input (tokens as input)
## Lambda layer (function that embeds input tokens: {fasttext, elmo, bert})
### elmo maybe just concat?
### if x-lingual: that function also maps
## (lstm layer)
## (timedistributed?) dense layer (softmax, dim=n_categories)

# FIT MODEL
# EVAL MODEL



def encode_cat(category):
    # 0 == O (none)
    # 1 == PER
    # 2 == LOC
    # 3 == ORG
    y = np.zeros(4)
    if 'PER' in category:
        y[1] = 1
    elif 'LOC' in category:
        y[2] = 1
    elif 'ORG' in category:
        y[3] = 1
    else:
        y[0] = 1
    return y

def load_data(input_file):
    with open(input_file, "r", encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
        next(csvreader)
        sentence = []
        labels = []
        X = []
        Y = []
        sentence_id = -1
        for row in csvreader:
            if row[1] != sentence_id:
                if len(sentence) > 0:
                    X.append(sentence)
                    Y.append(labels)
                sentence_id = row[1]
                sentence = [str(row[2])]
                labels = [encode_cat(row[3])]
            else:            
                sentence.append(str(row[2]))
                labels.append(encode_cat(row[3]))
    return X,Y

def pad_labels(labels):
    max_seqlen = max(len(s) for s in labels)
    lab0 = np.full((len(labels), max_seqlen, 4), fill_value=-999.)
    for x,label in enumerate(labels):
        seqlen = np.shape(label)[0]
        lab0[x, 0:seqlen, :] = label
    return lab0

def embed_fasttext(sentences, embeddings, predict, args):
    max_seqlen = max(len(s) for s in sentences) if sentences else 0
    if predict:
        max_seqlen = max(max_seqlen,256)
    if max_seqlen == 0:
        return []
    embedded = np.full((len(sentences), max_seqlen, args.dim), fill_value=-999.)
    #embedded = []
    for x,sentence in enumerate(sentences):
        seqlen = np.shape(sentence)[0]
        embedded[x, 0:seqlen, :] = np.array([embeddings[w] if w in embeddings else np.zeros(args.dim) for w in sentence])
        #seqlen = sentence[0].shape[0]
        #emb0[x, 0:seqlen, :] = sentence[0]
    #for s in sentences:
    #    embedded.append([embeddings[w] if w in embeddings else np.zeros(300) for w in s])
    #embedded = tf.map_fn(lambda s: tf.map_fn(lambda x: embeddings[x], s, dtype=tf.float32), sentences)
   
        
    return embedded

def load_fasttext(emb_file):
    embeddings = {}
    with open(emb_file, 'r') as embs:
        embs.readline()
        for line in embs:
            line = line.strip().split()
            try:
                embeddings[line[0]] = np.array([float(i) for i in line[1:]])
            except:
                continue
    #keys = list(embeddings.keys())
    #values = [embeddings[k] for k in keys]
    #table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(keys, values, value_dtype="float"))
    return embeddings
    
def generate_batch_data(inputfile, batch_size, embeddings, args, predict=False):
    #elmo = ElmoEmbedder(args.options, args.weights, -1)
    
    while True: # it needs to be infinitely iterable 
        x,y = load_data(inputfile)
        sys.stderr.write("INPUT SIZES X AND Y" + str(len(x)) + "  " + str(len(y))+"\n")
        assert len(x) == len(y)
        newxval = []
        yval = []
        for i in range(len(y)):
            newxval.append(x[i])
            yval.append(y[i])
            assert len(newxval) == len(yval)
            if i > 0 and i % batch_size == 0:
                xval0 = embed_fasttext(newxval, embeddings, predict, args)
                ypadded = pad_labels(yval)
                yield (np.array(xval0), np.array(ypadded))
                newxval = []
                yval = []
        if len(newxval) > 0:
            xval0 = embed_fasttext(newxval, embeddings, predict, args)
            ypadded = pad_labels(yval)
            yield (np.array(xval0), np.array(ypadded))

    
    
def main():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True)
    parser.add_argument("--eval_file", default=None, type=str, required=True)
    parser.add_argument("--embeddings_train", "--etr", default=None, type=str, required=True, help="Embeddings (.txt) of train language")
    parser.add_argument("--embeddings_eval", "--eev", default=None, type=str, required=False, help="Embeddings (.txt) of eval language, if different than train")
    parser.add_argument("--bs", default=64, type=int, help="batch size")
    parser.add_argument("--epoch", default=5, type=int)
    parser.add_argument("--experiment", default='default_experiment', type=str)
    parser.add_argument("--train_len", type=int)
    parser.add_argument("--eval_len", type=int)
    parser.add_argument("--dim", type=int, default=300)
    args = parser.parse_args()
    
    xlingual = args.embeddings_eval
    sys.stderr.write("****** LOADING EMBEDDINGS ******\n")
    embeddings = load_fasttext(args.embeddings_train)
    if xlingual:
        eval_embs = load_fasttext(xlingual)
    else:
        eval_embs = embeddings
    
    
    # LOAD DATA
    #x_tr, y_tr = load_data(args.train_file)
    #x_ev, y_ev = load_data(args.eval_file)
    
    #max_len = max(max(map(lambda x: len(x), x_tr)), max(map(lambda x: len(x), x_ev)))
    
    #raw_tr = embed_fasttext(x_tr, ft_embs)
    #if xlingual:
    #    raw_ev = embed_fasttext(x_ev, eval_embs)
    #else:
    #    raw_ev = embed_fasttext(x_ev, ft_embs)
        
    #x_tr = tf.keras.preprocessing.sequence.pad_sequences(raw_tr, maxlen=max_len, value=0, padding='post', dtype='float32')
    #x_ev = tf.keras.preprocessing.sequence.pad_sequences(raw_ev, maxlen=max_len, value=0, padding='post', dtype='float32')
    #y_tr = tf.keras.preprocessing.sequence.pad_sequences(y_tr, maxlen=max_len, value=np.array([1,0,0,0]), padding='post', dtype='float32')
    #y_ev = tf.keras.preprocessing.sequence.pad_sequences(y_ev, maxlen=max_len, value=np.array([1,0,0,0]), padding='post', dtype='float32')
    #x_tr = np.array(raw_tr)
    #x_ev = np.array(raw_ev)
    #y_tr = np.array(pad_labels(y_tr))
    #y_ev = np.array(pad_labels(y_ev))
    
    # NN
    #input_text = Input(shape=(max_len,), dtype="string")
    #embedding = Lambda(embed, output_shape=(max_len, 300)) (input_text)
    
    #embedding = Input(shape=(max_len,300), dtype="float32")
    input = Input(shape=(None,args.dim), dtype="float32")
    masking_layer = Masking(mask_value=-999., input_shape=(None,args.dim)) (input)
    lstm1 = Bidirectional(LSTM(units=2048, return_sequences=True)) (masking_layer)
    lstm2 = Bidirectional(LSTM(units=2048, return_sequences=True)) (lstm1)    
    out = TimeDistributed(Dense(4, activation="softmax")) (lstm2)
    #crf = CRF(5)
    #crf_out = crf(out)
    adam = optimizers.Adam(lr=1e-4, decay=1e-5)
    model = Model(input, out)
    sys.stderr.write("****** COMPILING MODEL ******\n")

    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
    #model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
    #model.summary()
    sys.stderr.write("****** STARTING TRAINING ******\n")

    model.fit_generator(generate_batch_data(args.train_file, args.bs, embeddings, args), steps_per_epoch=ceil(args.train_len/args.bs), epochs=args.epoch, validation_data=generate_batch_data(args.eval_file, args.bs, eval_embs, args), validation_steps=ceil(args.eval_len/args.bs) )

     
    #model.fit(x_tr, y_tr, epochs=args.epoch, batch_size=args.bs, validation_data=(x_ev, y_ev))
    #loss_and_metrics = model.evaluate(x_ev, y_ev, batch_size=64)
    #print(loss_and_metrics)
    sys.stderr.write("****** PREDICTING ******\n")

    y_predict = model.predict_generator(generate_batch_data(args.eval_file, args.bs, eval_embs, args, predict=True), steps=ceil(args.eval_len/args.bs))
    #model.save(args.save)
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

    with open(args.experiment+'.txt', 'w') as results:
        #results.write('confusion matrix\n')
        #cm = confusion_matrix(y_ev_i, y_pr_i)
        #results.write('\n'.join([' '.join(str(i)) for i in cm])+'\n\n')
        #results.write('micro f1 score\n')
        #results.write(str(f1_score(y_ev_i, y_pr_i, average='micro'))+'\n')
        results.write('macro f1 score: ')
        results.write(str(f1_score(y_ev_i, y_pr_i, average='macro'))+'\n')
        results.write('per category f1 score\n')
        results.write(' '.join([str(i) for i in f1_score(y_ev_i, y_pr_i, average=None)])+'\n')
    #print(confusion_matrix(y_ev_i, y_pr_i))
    #print(f1_score(y_ev_i, y_pr_i, average='micro'))
    #print(f1_score(y_ev_i, y_pr_i, average='macro'))
    #print(f1_score(y_ev_i, y_pr_i, average=None))

if __name__ == "__main__":
    main()    

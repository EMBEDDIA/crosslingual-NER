# Copyright (C) 2019  Matej Ulčar <matej.ulcar@fri.uni-lj.si>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from numpy.random import seed
seed(3)
from tensorflow import set_random_seed
set_random_seed(3)
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, LSTM, Input, TimeDistributed, Masking
from keras import optimizers
import keras.backend as K
import csv
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix, f1_score
from allennlp.commands.elmo import ElmoEmbedder


def encode_cat(category):
    # 0 == O (none/misc)
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
                if len(sentence) > 0 and len(sentence) < 1000:
                    X.append(sentence)
                    Y.append(labels)
                sentence_id = row[1]
                sentence = [str(row[2])]
                labels = [encode_cat(row[3])]
            else:            
                sentence.append(str(row[2]))
                labels.append(encode_cat(row[3]))
    return X,Y


def embed_elmo(sentences, elmo_embedder, xlingual):
    emb_batch = 256 # can cause high ram usage, reduce if necessary
    
    emb0 = []
    emb1 = []
    emb2 = []
    emba = []
    for s in range(0,len(sentences),emb_batch):
        if s%(emb_batch*8) == 0:
            print(s, len(sentences), s/len(sentences))
        emb = elmo_embedder.embed_batch(sentences[s:s+emb_batch])
        for sentence in emb:
            emba.append(np.mean(sentence, axis=0))
    embedded = [emba]
    if xlingual:
        embedded = apply_mapping(embedded)
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
    return embeddings
    
def apply_mapping(embedded_sentences):
    pass
    
    
def main():

    parser = argparse.ArgumentParser()

    # parameters
    parser.add_argument("--train_file", default=None, type=str, required=True, help="Path to NER train data .tsv")
    parser.add_argument("--eval_file", default=None, type=str, required=True, help="Path to NER test data .tsv")
    parser.add_argument("--options", default=None, type=str, required=True, help="Path to ELMo options.json file.")
    parser.add_argument("--weights", default=None, type=str, required=True, help="Path to ELMo weights.hdf5 file.")
    parser.add_argument("--bs", default=64, type=int, help="batch size")
    parser.add_argument("--epoch", default=3, type=int, help="number of epochs to train for")
    parser.add_argument('--cpu', action='store_true', help='Use cpu, instead of gpu to embed sentences.')
    parser.add_argument('--runs', default=5, type=int, help='Number of times to repeat training and evaluation.')
    args = parser.parse_args()
    if args.cpu:
        cuda_rank = -1
    else:
        cuda_rank = 0

    xlingual = False
    elmo = ElmoEmbedder(args.options, args.weights, cuda_rank)
    
    # LOAD DATA
    x_tr, y_tr = load_data(args.train_file)
    x_ev, y_ev = load_data(args.eval_file)
    
    max_len = max(max(map(lambda x: len(x), x_tr)), max(map(lambda x: len(x), x_ev)))
    
    raw_tr = embed_elmo(x_tr, elmo, xlingual)
    raw_ev = embed_elmo(x_ev, elmo, xlingual)
    
    x_tr0 = tf.keras.preprocessing.sequence.pad_sequences(raw_tr[0], maxlen=max_len, value=0, padding='post', dtype='float32')
    x_ev0 = tf.keras.preprocessing.sequence.pad_sequences(raw_ev[0], maxlen=max_len, value=0, padding='post', dtype='float32')
    y_tr = tf.keras.preprocessing.sequence.pad_sequences(y_tr, maxlen=max_len, value=np.array([1,0,0,0]), padding='post', dtype='float32')
    y_ev = tf.keras.preprocessing.sequence.pad_sequences(y_ev, maxlen=max_len, value=np.array([1,0,0,0]), padding='post', dtype='float32')
    
    y_tr = np.array(y_tr)
    y_ev = np.array(y_ev)
    
    # NN model
    input_layer = Input(shape=(max_len,1024), dtype="float32")
    masking_layer = Masking(mask_value=np.zeros(1024), input_shape=(max_len,1024)) (input_layer)
    lstm = LSTM(units=128, return_sequences=True, recurrent_dropout=0.1, dropout=0.1) (masking_layer)
    out = TimeDistributed(Dense(4, activation="softmax")) (lstm)
    adam = optimizers.Adam(lr=0.01, decay=1e-5)
    
    for run in range(args.runs): # we train and eval the model args.runs times
        model = Model(inputs=input_layer, outputs=out)
        model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(x_tr0, y_tr, epochs=args.epoch, batch_size=args.bs, validation_data=(x_ev0, y_ev))
        y_predict = model.predict(x_ev0, batch_size=args.bs)

        y_ev_i = []
        y_pr_i = []
        for s in range(len(y_ev)):
            for w in range(len(y_ev[s])):
                y_ev_i.append(np.argmax(y_ev[s][w]))
                y_pr_i.append(np.argmax(y_predict[s][w]))

        
        print('---***',run,'***---')
        print(confusion_matrix(y_ev_i, y_pr_i))
        print(f1_score(y_ev_i, y_pr_i, average='micro'))
        print(f1_score(y_ev_i, y_pr_i, average='macro'))
        print(f1_score(y_ev_i, y_pr_i, average=None))
if __name__ == "__main__":
    main()    

import csv
import numpy as np
from extra.apply_vecmap_transform import vecmap

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


#def embed_fasttext(sentences, embeddings, xlingual):
#    embedded = []
#    for s in sentences:
#        embedded.append([embeddings[w] if w in embeddings else np.zeros(300) for w in s])
#    #embedded = tf.map_fn(lambda s: tf.map_fn(lambda x: embeddings[x], s, dtype=tf.float32), sentences)
#    if xlingual:
#        embedded = apply_mapping(embedded)
#        
#    return embedded

def pad_labels(labels):
    max_seqlen = max(len(s) for s in labels)
    lab0 = np.full((len(labels), max_seqlen, 4), fill_value=-999.)
    for x,label in enumerate(labels):
        seqlen = np.shape(label)[0]
        lab0[x, 0:seqlen, :] = label
    return lab0

def embed_elmo(sentences, elmo_embedder, xlingual, normal=False, lang='', method='vecmap'):
    emb_batch = 256 #128 for et, 2<=n<8 for sv, en=?, others can use higher probably
    #swedish has problem around sentences 1500-1800 in train (extra high ram usage)
    #embedded = map(elmo_embedder.embed_sentence, sentences)
    if method=='vecmap':
        apply_mapping = apply_vecmap
    elif method=='muse':
        apply_mapping = apply_muse
    else:
        raise ValueError("Unsupported mapping method, use vecmap or muse.")
    max_seqlen = max(len(s) for s in sentences) if sentences else 0
    if max_seqlen == 0:
        return []
    max_seqlen = max(max_seqlen, 256)
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

def embed_elmogan(sentences, elmo_embedder, xlingual, args):
    emb_batch = 256 #128 for et, 2<=n<8 for sv, en=?, others can use higher probably
    #swedish has problem around sentences 1500-1800 in train (extra high ram usage)
    
    max_seqlen = max(len(s) for s in sentences) if sentences else 0
    if max_seqlen == 0:
        return []
    max_seqlen = 256
    emb0 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
    emb1 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
    emb2 = np.full((len(sentences), max_seqlen, 1024), fill_value=-999.)
    emba = []
    emb = elmo_embedder.embed_batch(sentences)
    for x,sentence in enumerate(emb):
        seqlen = sentence[0].shape[0]
        emb0[x, 0:seqlen, :] = elmogan_mapping(sentence[0], xlingual[0], args)
        emb1[x, 0:seqlen, :] = elmogan_mapping(sentence[1], xlingual[1], args)
        emb2[x, 0:seqlen, :] = elmogan_mapping(sentence[2], xlingual[2], args)

    embedded = [emb0, emb1, emb2]

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
    

def apply_vecmap(sentence, W, lang):
    if W:
        mapped_sentence = vecmap(sentence, W[lang], W['s'])
    else:
        mapped_sentence = sentence
    return mapped_sentence

def apply_muse(sentence, W):
    if W:
        mapped_sentence = np.array([np.matmul(W,v) for v in sentence])
    else:
        mapped_sentence = sentence
    return mapped_sentence

def elmogan_mapping(sentence, W, args):
    if W:
        if args.direction == 0:
            input = [sentence, sentence]
            mapped_sentence, _ = W.predict(input)
        else:
            input = [sentence, sentence]
            _, mapped_sentence = W.predict(input)
    else:
        mapped_sentence = sentence
    if args.normalize:
        normalize(mapped_sentence)
    return mapped_sentence

def normalize(matrix):
    def unit_normalize(matrix):
        norms = np.sqrt(np.sum(matrix**2, axis=1))
        norms[norms == 0] = 1
        matrix /= norms[:, np.newaxis]
    def center_normalize(matrix):
        avg = np.mean(matrix, axis=0)
        matrix -= avg
    unit_normalize(matrix)
    center_normalize(matrix)
    unit_normalize(matrix)

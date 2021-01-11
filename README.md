# crosslingual-NER
FastText and ELMo embeddings evaluations on NER task (monolingual and crosslingual)

If evaluating in a crosslingual setting, for vecmap mapping, apply the mapping (ie. crosslingual mode) already during the training, for other methods of mapping, train a normal monolingual model. For all methods, apply mapping during evaluation. For vecmap mapping, use matrices produced with `map_with_matrix.py` script from EMBEDDIA/vecmap-changes https://github.com/EMBEDDIA/vecmap-changes repository.

`ner_fasttext.py` : train and evaluate a NER model, using FastText embeddings (monolingual or crosslingual)
`train_elmo.py` : train a NER model, using ELMo embeddings (monolingual or crosslingual)
`predict_elmo.py` : evaluate a trained NER model from above (monolingual or crosslingual with vecmap)
`predict_muse.py` : evaluate a trained NER model crosslingualy, using MUSE mapping method
`predict_elmogan.py` : evaluate a trained NER model crosslingualy, using ELMoGAN mapping method https://github.com/EMBEDDIA/elmogan
`train_efml.py` : train a NER model, using ELMoForManyLangs embeddings (monolingual or crosslingual)
`predict_efml.py` : evaluate a trained NER model, using ELMoForManyLangs embeddings (monolingual or crosslingual with vecmap)


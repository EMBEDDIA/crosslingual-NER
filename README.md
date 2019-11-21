# crosslingual-NER
FastText and ELMo embeddings evaluations on NER task (monolingual and crosslingual)

There are three scripts to evaluate embeddings on NER task.

`ner_fasttext.py` : train and evaluate a NER model, using FastText embeddings (monolingual or crosslingual)
`ner_elmo_monolingual.py` : train and evaluate a NER model, using ELMo embeddings (monolingual)
`ner_elmo_crosslingual.py` : train and evaluate a NER model, using ELMo embeddings (crosslingual), mapped using matrices produced with `map_with_matrix.py` script from EMBEDDIA/vecmap-changes (https://github.com/EMBEDDIA/vecmap-changes) repository

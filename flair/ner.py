import flair

from typing import List
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings, TransformerWordEmbeddings

from flair.data import Corpus
from flair.datasets import ColumnCorpus


#https://github.com/flairNLP/flair/tree/master/examples/ner

def main():
    columns = {0: 'text', 1: 'ner'}
    corpus: Corpus = ColumnCorpus('../data/', columns,
                                  train_file='proximity_care_train.conll',
                                  test_file='proximity_care_test.conll',
                                  dev_file='proximity_care_dev.conll', column_delimiter='\t')

    print(len(corpus.train))
    print(corpus.dev[0].to_tagged_string('ner'))
    print(corpus.dev[0].to_tagged_string('text'))
    return

    tag_type = 'ner'
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

    # embedding_types: List[TokenEmbeddings] = [
    #     WordEmbeddings('glove'),
    #     #FlairEmbeddings('news-forward'),
    #     #FlairEmbeddings('news-backward'),
    # ]

    # embeddings = TransformerWordEmbeddings(
    #     model='dbmdz/bert-base-italian-cased',
    #     layers=-1,
    #     subtoken_pooling='first',
    #     fine_tune=True,
    #     use_context=0,
    #     respect_document_boundaries=False,
    # )

    embeddings = TransformerWordEmbeddings('dbmdz/bert-base-italian-cased')
    
    #embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)
                                        #use_rnn=False,
                                        #reproject_embeddings=False)

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.train('model/conllpp',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=5,
              embeddings_storage_mode='gpu')


if __name__ == '__main__':
    main()

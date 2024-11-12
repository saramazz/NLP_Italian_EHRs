import sys

from flair.nn import Classifier
from flair.data import Sentence
from flair.data import Corpus
from flair.datasets import ColumnCorpus, CSVClassificationCorpus


import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.ERROR
)

def main():

    model = sys.argv[1] if len(sys.argv) > 1 else 'model_run_ner/final-model.pt'
    
    sentences = []
    sentence = []
    for l in sys.stdin:
        l = l.strip()
        if l:
            sentence.append(l.split()[0])
        else:
            sentences.append(Sentence(' '.join(sentence), use_tokenizer=False))
            sentence = []


    cl  = Classifier.load(model)

    cl.predict(sentences)

    for sentence in sentences:
        idx2label = {}
        for t in sentence.get_labels():
            is_b = True
            for tok in t.data_point.tokens:
                idx2label[tok.idx] = f'B-{t.value}' if is_b else f'I-{t.value}'
                is_b = False
                
        #labels = [sentence.get_labels()]
        #print(labels)
        #print(sentence)
        for token in sentence:
            print(f'{token.text}\t{idx2label.get(token.idx, "O")}')
        print()

if __name__ == '__main__':
    main()

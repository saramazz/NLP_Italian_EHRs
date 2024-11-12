import sys
import spacy
from spacy.tokens import DocBin
from spacy.training import offsets_to_biluo_tags

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else 'dbmdz/model-best'
    corpus = sys.argv[2] if len(sys.argv) > 1 else 'predict_dbmdz.spacy'
    nlp = spacy.load(model)
    docbin = DocBin().from_disk(corpus)

    for doc in docbin.get_docs(nlp.vocab):
        for token in doc:
            iob = f'{token.ent_iob_}-{token.ent_type_}' if token.ent_type_ else token.ent_iob_
            print(f'{token.text}\t{iob}')
        print()
        
if __name__ == '__main__':
    main()

#!/usr/bin/env python

import sys
import json
import stanza

class ProximityCareTokenizer:
    def __init__(self, lang='it', expand_mwt=False):
        self.nlp = stanza.Pipeline(lang=lang, processors='tokenize,mwt', use_gpu=False)
        self.expand_mwt = expand_mwt

    @staticmethod
    def extract_annotations(annotations):
        for ann in annotations:
            for result in ann['result']:
                yield result['value']

    @staticmethod
    def is_overlap(interval1, interval2):
        return max(0, min(interval1[1], interval2[1]) - max(interval1[0], interval2[0])) > 0
        
    def tokenize_json(self, json_file):
        with open(json_file, encoding='utf-8') as fin:  # Specify the encoding here
            data = json.load(fin)
            for anamnesis in data:

                print(f"-DOCSTART-\t{anamnesis['data']['ID']}")
                
                annotations = [ann for ann in self.extract_annotations(anamnesis['annotations'])]

                text = anamnesis['data']['anamnesis']
                doc = self.nlp(text)
                for i, sentence in enumerate(doc.sentences):
                    iob = 'O'
                    for token in sentence.tokens:
                        if self.expand_mwt:
                            for w in token.words:
                                label = None
                                for ann in annotations:
                                    if self.is_overlap((token.start_char, token.end_char), (ann['start'], ann['end'])):
                                        assert len(ann['labels']) == 1, ann['labels']
                                        label = ann['labels'][0]
                                        iob = f'B-{label}' if iob == 'O' else f'I-{label}'
                                        break
                                else:
                                    iob = 'O'

                                print(f'{w.text}\t{iob}')
                        else:
                            for ann in annotations:
                                if self.is_overlap((token.start_char, token.end_char), (ann['start'], ann['end'])):
                                    assert len(ann['labels']) == 1, ann['labels']
                                    label = ann['labels'][0]
                                    iob = f'B-{label}' if (iob == 'O') or (label != iob[2:]) else f'I-{label}'
                                    break
                            else:
                                iob = 'O'
                            
                            print(f'{token.text}\t{iob}')

                    print()

    def tokenize(self, sentences):
        doc = self.nlp(sentences)
        for i, sentence in enumerate(doc.sentences):
            print(*[f'{token.text}' for token in sentence.tokens], sep='\n')
            print()



def main():
    tokenizer = ProximityCareTokenizer()
    filename = sys.argv[1]
    if filename.endswith('.json'):
        tokenizer.tokenize_json(sys.argv[1])
    elif filename.endswith('.txt'):
        with open(filename) as finput:
            doc = []
            prev_docstart = None
            for line in finput:
                line = line.strip()
                if line.startswith('-DOCSTART-'):
                    if prev_docstart:
                        print(prev_docstart)
                        tokenizer.tokenize(tokenizer.nlp(' '.join(doc)))
                    doc = []
                    prev_docstart = line
                    continue
                doc.append(line)

        if prev_docstart:
            print(prev_docstart)
            tokenizer.tokenize(tokenizer.nlp(' '.join(doc)))
                    
def main2():

    nlp = stanza.Pipeline(lang='it', processors='tokenize', use_gpu=False)
    s = ' '.join([s for s in sys.stdin])
    doc = nlp(s)
    for i, sentence in enumerate(doc.sentences):
        print(*[f'{token.text} {token}' for token in sentence.tokens], sep='\n')
        print()


if __name__ == '__main__':
    main()

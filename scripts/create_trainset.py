#!/usr/bin/env python

import sys
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 82

class SplitDataset:
    def __init__(self, input):
        self.input = input

    def collect(self):
        sentences = {}
        sentence = []
        for line in self.input:
            line = line.strip()
            if line:
                if line.startswith('-DOCSTART-'):
                    continue
                sentence.append(line)
            else:
                if sentence:
                    key = ' '.join([d.split()[0] for d in sentence])
                    if key not in sentences:
                        sentences[key] = sentence
                sentence = []
        return list(sentences.values())

    def split(self, sentences, test_size=0.2):
        train, test = train_test_split(sentences, test_size=test_size, random_state=SEED)
        return train, test

    def print_out(self, sentences, fname):
        with open(fname, 'w') as fout:
            for sentence in sentences:
                for line in sentence:
                    print('\t'.join(line.split()), file=fout)
                print('', file=fout)

class StratifiedSplitDataset(SplitDataset):
    def __init__(self, input, external_csv):
        super().__init__(input)
        self.df = pd.read_csv(external_csv)

    def collect(self):
        sentence = []
        report_id = None
        reports = {}
        for line in self.input:
            line = line.strip()
            if line:
                if line.startswith('-DOCSTART-'):
                    report_id = int(float(line.split()[-1]))
                    reports[report_id] = []
                sentence.append(line)
            else:
                if sentence:
                    reports[report_id].append(sentence)
                sentence = []
        return reports

    def split(self, reports, id_field='CODINT', stratify_field='Amylo'):
        self.df = self.df.dropna(subset=[id_field, stratify_field])
        X_train, X_test= train_test_split(self.df, stratify=self.df[stratify_field], test_size=0.20, random_state=SEED)
        X_train, X_dev = train_test_split(X_train, stratify=X_train[stratify_field], test_size=0.20, random_state=SEED)
        
        return [reports[int(el)] for el in X_train[id_field] if int(el) in reports], [reports[int(el)] for el in X_dev[id_field] if int(el) in reports], [reports[int(el)] for el in X_test[id_field] if int(el) in reports]

    def print_out(self, reports, fname):
        with open(fname, 'w') as fout:
            for report in reports:
                for sentence in report:
                    for line in sentence:
                        print('\t'.join(line.split()), file=fout)
                    print('', file=fout)

def main():
    if sys.argv[1] == 'stratified':
        splitter = StratifiedSplitDataset(sys.stdin, 'data/Casi_Controlli.csv')
        sentences = splitter.collect()
        train, dev, test = splitter.split(sentences)
        splitter.print_out(train, 'train.conll')
        splitter.print_out(dev, 'dev.conll')
        splitter.print_out(test, 'test.conll')
    else:
        splitter = SplitDataset(sys.stdin)
        sentences = splitter.collect()
        train, test = splitter.split(sentences)
        train, dev = splitter.split(train)
        splitter.print_out(train, 'train.conll')
        splitter.print_out(dev, 'dev.conll')
        splitter.print_out(test, 'test.conll')


if __name__ == '__main__':
    main()

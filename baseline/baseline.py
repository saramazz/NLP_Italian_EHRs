import re
import os
import json
import sys


def create_rss():
    dictionary = {}
    for line in sys.stdin:
        line = line.strip()
        if line:
            form, tag = line.split()
            if tag != 'O':
                if tag not in dictionary:
                    dictionary[tag] = set()
                dictionary[tag].add(form.lower())
    for k in dictionary.keys():
        dictionary[k] = list(dictionary[k])

    with open('baseline_dictionary.json', 'w') as fout:
        json.dump(dictionary, fout)

def create_rss_v2():
    dictionary = {}
    tags = []
    for line in sys.stdin:
        line = line.strip()
        if line:
            form, tag = line.split()
            if tag != 'O':
                if tag.startswith('B'):
                    if tags:
                        k = ' '.join([x[0].lower() for x in tags])
                        if not tags[0][1] in dictionary:
                            dictionary[tags[0][1]] = set()
                        dictionary[tags[0][1]].add(k)
                        tags = []
                    tags.append((form, tag))
                else:
                    assert len(tags) > 0, line
                    tags.append((form, tag))
                    
    for k in dictionary.keys():
        dictionary[k] = list(dictionary[k])

    with open('baseline_dictionary_v2.json', 'w') as fout:
        json.dump(dictionary, fout)


def find_subsequence_indices(tokens, subsequence):
    n, m = len(tokens), len(subsequence)
    for i in range(n - m + 1):
        if tokens[i:i + m] == subsequence:
            return i, i + m - 1
    return None

def is_overlapping(r1, r2):
    return not (r1[1] < r2[0] or r2[1] < r1[0])

def predict(rev_dictionary, input_file, out_file):
    with open(input_file) as fin:
        with open(out_file, 'w') as fout:
            sentence = []
            tags = []
            golds = []
            for line in fin:
                line = line.strip()
                if line:
                    form, gold = line.split()
                    form = form.lower()
                    sentence.append(form)
                    golds.append(gold)
                    tags.append('O')
                else:                    
                    occupied = []
                    for k in rev_dictionary:
                        r = find_subsequence_indices(sentence, k.split(' '))
                        if r:
                            if any(is_overlapping(r, occ) for occ in occupied):
                                continue
                            tag = rev_dictionary[k].split('-')[-1]
                            t = f'B-{tag}'
                            for i in range(r[0], r[1]+1):
                                tags[i] = t
                                t = f'I-{tag}'
                            occupied.append(r)

                    for f,g,t in zip(sentence, golds, tags):
                        print(f'{f}\t{g}\t{t}', file=fout)
                    sentence = []
                    tags = []
                    golds = []
                    print(file=fout)
                    
                    
                    
def predict_v0(rev_dictionary, input_file, out_file):
    with open(input_file) as fin:
        with open(out_file, 'w') as fout: 
            for line in fin:
                line = line.strip()
                if line:
                    form, gold = line.split()
                    pred = 'O' if form.lower() not in rev_dictionary else rev_dictionary[form.lower()]
                    print(f'{form}\t{gold}\t{pred}', file=fout)
                else:
                    print(line, file=fout)


def main():
    if not os.path.exists('baseline_dictionary.json'):
        create_rss()
    if not os.path.exists('baseline_dictionary_v2.json'):
        create_rss_v2()
    with open('baseline_dictionary_v2.json') as fin:
        dictionary = json.load(fin)
    rev_dictionary = {}
    for tag, values in dictionary.items():
        for v in values:
            rev_dictionary[v] = tag

    sorted_dict = dict(sorted(rev_dictionary.items(), key=lambda item: len(item[0]), reverse=True))
    
    predict(sorted_dict, '../data/proximity_care_dev.conll', 'baseline_output_dev.conll')
    predict(sorted_dict, '../data/proximity_care_test.conll', 'baseline_output_test.conll')

if __name__ == '__main__':
    main()

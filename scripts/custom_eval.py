import sys
from collections import Counter
from functools import cmp_to_key
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


def evaluate():
    true_labels = []
    predict_labels = []

    for line in sys.stdin:
        fields = line.strip().split()
        if fields:
            predict_labels.append(fields[-1])
            true_labels.append(fields[-2])
    
    counter = Counter(true_labels)
    display_labels = []
    for l, _ in counter.most_common():
        print(l, l[2:])
        if l == 'O':
            display_labels.append(l)
        elif 'B-{}'.format(l[2:]) not in display_labels:
            display_labels.append('B-{}'.format(l[2:]))
            display_labels.append('I-{}'.format(l[2:]))

    print(classification_report(true_labels, predict_labels, labels=display_labels))

    cm = confusion_matrix(true_labels, predict_labels, labels=display_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(30,30))
    disp.plot(ax=ax)
    plt.savefig('confusion_matrix_test.png')

if __name__ == '__main__':
    evaluate()

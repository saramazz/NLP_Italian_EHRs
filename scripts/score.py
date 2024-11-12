import sys
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix

from nerval import plot_confusion_matrix, crm, get_clean_entities

pd.set_option('display.max_rows', 500)


def main():
    fname = sys.argv[1]

    df = pd.read_csv(fname, sep='\t', quoting=3)
    df.test = df.test.astype(str)
    y_true = df.test
    for k in ('multiconer', 'flair', 'spacy'):
        df[k] = df[k].astype(str)
        y_pred = df[k]
        print(k)
        print(classification_report(y_pred=y_pred, y_true=y_true, labels=df.test.unique()))
        print()
        #cm = confusion_matrix(y_pred=y_pred, y_true=y_true, labels=df.test.unique())
        #print(pd.DataFrame(cm, columns=df.test.unique()))
        cr, cm, cm_labels = crm(y_true, y_pred, scheme='BIO')
        print(cr)
        #print(cm)
        print()


if __name__ == '__main__':
    main()

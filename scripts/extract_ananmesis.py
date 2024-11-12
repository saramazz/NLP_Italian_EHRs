import sys
import csv
import pandas as pd
from ner.tokenizer.tokenizer import ProximityCareTokenizer

def main(filename):
    tokenizer = ProximityCareTokenizer()
    df = pd.read_csv(filename, sep='\t', encoding='ISO-8859-1')
    for r in df.REFERTO:
        tokenizer.tokenize(r)

if __name__ == "__main__":
    main(sys.argv[1])

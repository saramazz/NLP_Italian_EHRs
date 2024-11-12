import sys
import spacy
from spacy.tokens import Doc

class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        spaces = [True] * len(words)
        # Avoid zero-length tokens
        for i, word in enumerate(words):
            if word == "":
                words[i] = " "
                spaces[i] = False
        # Remove the final trailing space
        if words[-1] == " ":
            words = words[0:-1]
            spaces = spaces[0:-1]
        else:
           spaces[-1] = False

        return Doc(self.vocab, words=words, spaces=spaces)
        

def print_iob(sentence):
    for token in sentence:
        iob = f'{token.ent_iob_}-{token.ent_type_}' if token.ent_type_ else token.ent_iob_
        print(f'{token.text}\t{iob}')
    print()


def main():
    spacy.prefer_gpu()
    nlp = spacy.load(sys.argv[1])
    nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

    sentence = []
    for line in sys.stdin:
        line = line.strip()
        if line.startswith('-DOCSTART-'):
            if sentence:
                print_iob(nlp(' '.join(sentence)))
            print(line)
            sentence = []
        elif line == '' and sentence:
            print_iob(nlp(' '.join(sentence)))
            sentence = []
        else:
            sentence.append(line)
    if sentence:
        print_iob(nlp(' '.join(sentence)))

if __name__ == '__main__':
    main()

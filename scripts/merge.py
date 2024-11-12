import sys

def get_tag(tag_a, tag_b):
    tag = []
    if tag_a != 'O':
        tag.append(tag_a)
    if tag_b != 'O':
        tag.append(tag_b)
    if not tag:
        tag = ['O']
    return '|'.join(tag)

def main():
    # form_a pos_a lemma_a tag_a form_b pos_b lemma_b tag_b pred_a pred_b
    # Insufficienza   Sfs     insufficienza   B-DISO  Insufficienza   Sfs     insufficienza   O       B-DISO  O
    for l in sys.stdin:
        l = l.strip()
        if l:
            s = l.split('\t')
            if len(s) > 3:
                form_a, pos_a, lemma_a, tag_a, form_b, pos_b, lemma_b, tag_b, pred_a, pred_b = s
                for ab in ((form_a, form_b), (pos_a, pos_b), (lemma_a, lemma_b)):
                    assert ab[0] == ab[1]
                tag = get_tag(tag_a, tag_b)
                predicted = get_tag(pred_a, pred_b)
                print('\t'.join([form_a, tag, predicted]))
            else:
                form, pred_a, pred_b = s
                predicted = get_tag(pred_a, pred_b)
                print('\t'.join([form, predicted]))
        else:
            print()

if __name__ == '__main__':
    main()

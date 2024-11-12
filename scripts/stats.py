import sys

stats = {'reports': 0, 'sentences': 0, 'tokens': 0, 'chunks': 0}
for line in open(sys.argv[1]):
    if line.startswith('-DOCSTART-'):
        stats['reports'] += 1
        continue
    elif not line.strip():
        stats['sentences'] += 1
    stats['tokens'] += 1
    if line.strip():
        tag = line.strip().split()[-1]
        if tag != 'O' and tag.startswith('B-'):
            stats['chunks'] += 1
print(stats)

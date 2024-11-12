files = ['/content/project-1-at-2023-03-16-15-24-968466f0.json.conll', '/content/project-14-at-2023-03-16-10-15-b78e6265.json.conll', '/content/project-5-at-2023-03-21-13-17-d8cff330.json.conll']

#collect tokens 2 tag

def extract(entity):
  norm = ' '.join([t.lower() for t,e in entity])
  tag = [e[2:] for t,e in entity]
  assert len(set(tag)) == 1
  tag =  tag[0]
  return norm, tag

def collect_entities(files):
  tag2tokens = {}
  token2tags = {}
  for f in files:
    with open(f) as fin:
      entity = None
      for i, line in enumerate(fin):
        line = line.strip()
        if line:
          t, e = line.split()
          if e.startswith('I'):
            assert entity is not None, entity
            entity.append((t, e))
          else:
            if entity:
              token, tag = extract(entity)
              if tag not in tag2tokens:
                tag2tokens[tag] = set()
              tag2tokens[tag].add(token)
              if token not in token2tags:
                token2tags[token] = set()
              token2tags[token].add(tag)
            if e.startswith('B'):
              entity = [(t, e)]
            else:
              entity = None
  return tag2tokens, token2tags

tag2tokens, token2tags = collect_entities(files)


import json

files = ['/content/project-1-at-2023-03-16-15-24-968466f0.json.conll', '/content/project-14-at-2023-03-16-10-15-b78e6265.json.conll', '/content/project-5-at-2023-03-21-13-17-d8cff330.json.conll']

def find_entity_with_tag(jsonfile, entity, tag):
  with open(jsonfile) as fin:
    docs = json.load(fin)
    for doc in docs:
      annotations = doc['annotations']
      for annotation in annotations:
        for result in annotation['result']:
          if tag in result['value']['labels'] and (result['value']['text'].lower() in entity.lower() or entity.lower() in result['value']['text'].lower()):
            print(f"Found {entity} tagged with {tag}")
            print(f"{jsonfile} - {result}\n{doc['data']['anamnesis']}")
            print()
            break

entities_to_check = [{'entity': 'ipertensione arteriosa', 'tag': 'DY'}, {'entity': 'diabete', 'tag': 'NODY'}]
for entity_tag in entities_to_check:
  for file in files:
    find_entity_with_tag(file.replace('.conll', ''), **entity_tag)


import json

files = ['/content/project-1-at-2023-03-16-15-24-968466f0.json.conll', '/content/project-14-at-2023-03-16-10-15-b78e6265.json.conll', '/content/project-5-at-2023-03-21-13-17-d8cff330.json.conll']

def extract_anamnesis2entities(jsonfile):
  anamnesis2entities = []
  with open(jsonfile) as fin:
    docs = json.load(fin)
    for doc in docs:
      entities_with_tag = []
      annotations = doc['annotations']
      
      for annotation in annotations:
        for result in annotation['result']:
          entities_with_tag.append({'entity': result['value']['text'], 'tags': result['value']['labels']})
      anamnesis2entities.append({'anamnesis': doc['data']['anamnesis'], 'entities': entities_with_tag, 'file': jsonfile})
  return anamnesis2entities
  

def find_entity_without_tag(anamnesis2entities, entity, tags):
  for annotation in anamnesis2entities:
    if entity.lower() in annotation['anamnesis']:
      if entity.lower() not in [el['entity'].lower() for el in annotation['entities']]:
        print(f"{entity} | {annotation['file']} not tagged in \n{annotation}")
      else:
        for e in filter(lambda x: x['entity'].lower() == entity.lower(), annotation['entities']):
          if not set(tags).intersection(e['tags']):
            print(f"{entity} | {annotation['file']} has different tags {tags} vs {e['tags']} in \n{annotation}")

anamnesis2entities = []
for f in files:
  anamnesis2entities.extend(extract_anamnesis2entities(f.replace('.conll', '')))

for entity, tags in token2tags.items():
  find_entity_without_tag(anamnesis2entities, entity, tags)

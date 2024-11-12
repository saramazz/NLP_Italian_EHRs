### Risk factor ###
# train on spacy
# make GPU=3 DATA=risk_factor_data PROXIMITY_DATASET=../risk_factor_data OUTPUT_FOLDER=risk_factor_dbmz_202407 train


#CONFIG = config.cfg
CONFIG = config_dbmdz.cfg
OUTPUT_FOLDER = dbmdz_20230705_strat

PROXIMITY_DATASET = ../data
DATA = data

GPU=2

MODEL=$(OUTPUT_FOLDER)/model-best

$(DATA):
	mkdir -p $(DATA)

proximity_care_dev.spacy: $(DATA) #target to create  proximity_care_dev.spacy:
	python -m spacy convert $(PROXIMITY_DATASET)/proximity_care_dev.conll $<

proximity_care_train.spacy: $(DATA)
	python -m spacy convert $(PROXIMITY_DATASET)/proximity_care_train.conll $<

proximity_care_test.spacy: $(DATA)
	python -m spacy convert $(PROXIMITY_DATASET)/proximity_care_test.conll $<


train: proximity_care_dev.spacy proximity_care_train.spacy
	mkdir -p $(OUTPUT_FOLDER)
	export CUDA_VISIBLE_DEVICES="$(GPU)"; python -m spacy train $(CONFIG) --paths.train $(DATA)/proximity_care_train.spacy --paths.dev $(DATA)/proximity_care_dev.spacy --output ./$(OUTPUT_FOLDER) --gpu-id 0

eval:
	export CUDA_VISIBLE_DEVICES="$(GPU)";python -m spacy benchmark accuracy $(MODEL) $(DATA)/proximity_care_dev.spacy --output eval_$(OUTPUT_FOLDER) --gold-preproc --gpu-id 0


predict:
	export CUDA_VISIBLE_DEVICES="$(GPU)";python -m spacy apply $(MODEL) $(DATA)/proximity_care_dev.spacy predict_$(OUTPUT_FOLDER) --gpu-id 0

convert:
	python to_iob.py > predict_$(OUTPUT_FOLDER).iob

conlleval:
	cut -f2 predict_$(OUTPUT_FOLDER).iob | paste $(PROXIMITY_DATASET)/proximity_care_dev.conll - | python ../scripts/conlleval.py

# convert:
# 	export CUDA_VISIBLE_DEVICES="$(GPU)";python -m spacy convert predict_$(OUTPUT_FOLDER).spacy . -c conll

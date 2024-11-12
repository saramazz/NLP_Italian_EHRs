### Risk factor ###
# in order to create conll files into the risk_factor_data dir:
# make DATASET_DIR=risk_factor_data ANNOTATIONS_DIR=risk_factor_data/project-6-at-2024-06-04-09-24-da440647 json_to_conll proximity_care_dataset



DATASET_DIR=data
LAN=it

CORPUS=b

# TRAINSET=$(DATASET_DIR)/ris_train.$(CORPUS).conll
# DEVSET=$(DATASET_DIR)/ris_dev.$(CORPUS).conll
# TESTSET=$(DATASET_DIR)/ris_test.$(CORPUS).conll

TRAINSET=$(DATASET_DIR)/proximity_care_train.conll
DEVSET=$(DATASET_DIR)/proximity_care_dev.conll
TESTSET=$(DATASET_DIR)/proximity_care_test.conll

GPU=2
N_GPUS=1
EPOCHS=200

MULTICONER_ENCODER_MODEL=xlm-roberta-base
#MULTICONER_ENCODER_MODEL=dbmdz/bert-base-italian-cased

MULTICONER_MODEL_NAME=xlm_roberta_base_20230705_strat
#MULTICONER_MODEL_NAME=dbmdz_bert-base-italian-cased_20230613

MULTICONER_OUT_DIR=experiments
EVAL_DIR=eval

BATCH=64
LR=0.0001

MULTICONER_NER_DIR=multiconer-baseline

TAGSET = proximity_care #ris

ris_dataset:
	cat $(DATASET_DIR)/anamnesi.$(CORPUS).iob $(DATASET_DIR)/esami.$(CORPUS).iob |python scripts/create_trainset.py && mv train.conll $(TRAINSET) && mv dev.conll $(DEVSET) && mv test.conll $(TESTSET)

ANNOTATIONS_DIR=$(DATASET_DIR)/stratified_proximity_care_json_2023-07-05

json_to_conll:
	for f in $(ANNOTATIONS_DIR)/*json ; do \
		python tokenizer/tokenizer.py $$f > $$f.conll ; \
	done

proximity_care_dataset:
	cat $(ANNOTATIONS_DIR)/*.conll | python scripts/create_trainset.py stratified && mv train.conll $(TRAINSET).doc && mv dev.conll $(DEVSET).doc && mv test.conll $(TESTSET).doc
	grep -v "\-DOCSTART\-" $(TRAINSET).doc > $(TRAINSET)
	grep -v "\-DOCSTART\-" $(DEVSET).doc > $(DEVSET)
	grep -v "\-DOCSTART\-" $(TESTSET).doc > $(TESTSET)

$(TRAINSET).ids:
	grep "\-DOCSTART" $(TRAINSET).doc | cut -f2 > $@

$(DEVSET).ids:
	grep "\-DOCSTART" $(DEVSET).doc | cut -f2 > $@

$(TESTSET).ids:
	grep "\-DOCSTART" $(TESTSET).doc | cut -f2 > $@



### MULTICONER ###

$(MULTICONER_OUT_DIR):
	mkdir $@

$(MULTICONER_OUT_DIR)/$(MULTICONER_MODEL_NAME)_lr$(LR)_ep$(EPOCHS)_batch$(BATCH): $(MULTICONER_OUT_DIR)
	export CUDA_VISIBLE_DEVICES="$(GPU)"; python $(MULTICONER_NER_DIR)/train_model.py --iob_tagging $(TAGSET) --train $(TRAINSET) --dev $(DEVSET) --out_dir $@ --model_name $(MULTICONER_MODEL_NAME) --gpus $(N_GPUS) --epochs $(EPOCHS) --encoder_model $(MULTICONER_ENCODER_MODEL) --batch_size $(BATCH) --lr $(LR)

MULTICONER_MODEL_FILE = $(MULTICONER_OUT_DIR)/$(MULTICONER_MODEL_NAME)_lr$(LR)_ep$(EPOCHS)_batch$(BATCH)/$(MULTICONER_MODEL_NAME)/lightning_logs/version_0/

evaluate:
	mkdir -p $(EVAL_DIR)
	export CUDA_VISIBLE_DEVICES="$(GPU)"; python $(MULTICONER_NER_DIR)/evaluate.py --iob_tagging $(TAGSET) --test $(DEVSET) --out_dir $(EVAL_DIR) --gpus 1 --encoder_model $(MULTICONER_ENCODER_MODEL) --model $(MULTICONER_MODEL_FILE) --prefix $(MULTICONER_MODEL_NAME)_results

custom_evaluate:
	export CUDA_VISIBLE_DEVICES="$(GPU)"; python $(MULTICONER_NER_DIR)/predict_tags.py --iob_tagging $(TAGSET) --test $(DEVSET) --gpus 1 --encoder_model $(MULTICONER_ENCODER_MODEL) --model $(MULTICONER_MODEL_FILE) --prefix $(MULTICONER_MODEL_NAME)_results --max_length 500 --batch_size 8 --standard_output | paste $(DEVSET) - > eval_tmp_custom.conll
	python scripts/custom_eval.py < eval_tmp_custom.conll

predict:
	export CUDA_VISIBLE_DEVICES="$(GPU)"; python $(MULTICONER_NER_DIR)/predict_tags.py --iob_tagging $(TAGSET) --test $(TESTSET) --out_dir test --gpus 1 --encoder_model $(MULTICONER_ENCODER_MODEL) --model $(MULTICONER_MODEL_FILE) --prefix $(MULTICONER_MODEL_NAME)_results --max_length 500 --batch_size 8

data/amylo_anamnesi.tok:
	PYTHONPATH='..' python scripts/extract_ananmesis.py data/amylo_anamnesi.tsv > $@

annotate: data/amylo_anamnesi.tok
	mkdir -p amylo_anamnesi
	export CUDA_VISIBLE_DEVICES="$(GPU)"; python $(MULTICONER_NER_DIR)/predict_tags.py --iob_tagging $(TAGSET) --test $< --out_dir amylo_anamnesi --gpus 1 --encoder_model $(MULTICONER_ENCODER_MODEL) --model $(MULTICONER_MODEL_FILE) --prefix $(MODEL_NAME)_results --max_length 500 --batch_size 8


PREDICTED_FILE_A=test/xlmr_ner_corpus_a_results_base_xlmr_ner_corpus_a_timestamp_1674749030.2025368_final.tsv
PREDICTED_FILE_B=test/xlmr_ner_corpus_b_results_base_xlmr_ner_corpus_b_timestamp_1674748974.6853185_final.tsv

check:
	paste $(DATASET_DIR)/test.a.conll $(DATASET_DIR)/test.b.conll $(PREDICTED_FILE_A) $(PREDICTED_FILE_B) | python scripts/merge.py > merged.conll

sample.tok: sample.txt
	python ../tokenizer/tokenizer.py < $< > $@

sample.$(CORPUS).conll: sample.tok
	export CUDA_VISIBLE_DEVICES="$(GPU)"; python $(MULTICONER_NER_DIR)/predict_tags.py --iob_tagging ris --test $< --out_dir sample --gpus 1 --encoder_model $(MULTICONER_ENCODER_MODEL) --model $(MULTICONER_MODEL_FILE) --prefix $(MULTICONER_MODEL_NAME)_results --max_length 100 --batch_size 8

# paste sample.tok sample/xlmr_ner_corpus_a_results_base_xlmr_ner_corpus_a_timestamp_1674749030.2025368_final.tsv sample/xlmr_ner_corpus_b_results_base_xlmr_ner_corpus_b_timestamp_1674748974.6853185_final.tsv | python ../scripts/merge.py > sample.iob


%.output: %.iob
	cut -f2 $< > $@

%.form: %.conll
	cut -f1 $< > $@

## evaluation ###spacy

TO_EVALUATE=dev
#TO_EVALUATE=test
DATASET_TO_EVALUATE=$(DATASET_DIR)/proximity_care_$(TO_EVALUATE).form
DATASET_TO_COMPARE=$(DATASET_DIR)/proximity_care_$(TO_EVALUATE).conll
SPACY_DATASET_TO_EVALUATE=$(SPACY_DATA)/proximity_care_$(TO_EVALUATE).spacy



### multiconer ###

multiconer_$(TO_EVALUATE)_predict.iob: $(DATASET_TO_EVALUATE)
	export CUDA_VISIBLE_DEVICES="$(GPU)"; python $(MULTICONER_NER_DIR)/predict_tags.py --iob_tagging $(TAGSET) --test $(DATASET_TO_EVALUATE) --gpus 1 --encoder_model $(MULTICONER_ENCODER_MODEL) --model $(MULTICONER_MODEL_FILE) --prefix $(MULTICONER_MODEL_NAME)_results --max_length 500 --batch_size 8 --standard_output | paste $(DATASET_TO_EVALUATE) - > $@ 

multiconer_$(TO_EVALUATE)_conlleval: multiconer_$(TO_EVALUATE)_predict.iob
	#python scripts/conlleval.py < $< > $@
	cut -f2 $< | paste $(DATASET_TO_COMPARE) - | python scripts/conlleval.py > $@


### spacy ###

SPACY_DATA = spacy/data
SPACY_LM = dbmdz
#SPACY_MODEL = spacy/$(SPACY_LM)_20230705_strat/model-best
SPACY_MODEL = ../models/spacy/model-best


spacy_$(TO_EVALUATE)_predict_$(SPACY_LM).spacy:
	export CUDA_VISIBLE_DEVICES="$(GPU)";python -m spacy apply $(SPACY_MODEL) $(SPACY_DATASET_TO_EVALUATE) spacy_$(TO_EVALUATE)_predict_$(SPACY_LM) --gpu-id 0

spacy_$(TO_EVALUATE)_predict_$(SPACY_LM).iob: spacy_$(TO_EVALUATE)_predict_$(SPACY_LM).spacy
	python spacy/to_iob.py $(SPACY_MODEL) $< > $@

# cut -f2 spacy_test_predict_dbmdz.iob | paste data/proximity_care_test.conll - | sed -e "s/[[:space:]]\+/ /g" | perl scripts/conlleval.txt -l

spacy_$(TO_EVALUATE)_$(SPACY_LM)_conlleval: spacy_$(TO_EVALUATE)_predict_$(SPACY_LM).iob
	cut -f2 $< | paste $(DATASET_TO_COMPARE) - | python scripts/conlleval.py > $@


### flair ###

#FLAIR_MODEL=flair/flair_dbmdz_bert-base-italian-xxl-cased_20230705/final-model.pt
FLAIR_MODEL=flair/flair_dbmdz_bert-base-italian-xxl-cased_crf_pat10_ep200_20230705_strat/final-model.pt

CRF=_crf

flair$(CRF)_$(TO_EVALUATE)_predict.iob: 
	python flair/to_iob.py $(FLAIR_MODEL) < $(DATASET_TO_EVALUATE) > $@.tmp
	tail -n +2 $@.tmp > $@ # logging bug in flair
	rm $@.tmp

flair$(CRF)_$(TO_EVALUATE)_conlleval: flair$(CRF)_$(TO_EVALUATE)_predict.iob
	cut -f2 $< | paste $(DATASET_TO_COMPARE) - | python scripts/conlleval.py > $@


### general eval ###
proximity_$(TO_EVALUATE)_eval.tsv: multiconer_$(TO_EVALUATE)_predict.output flair$(CRF)_$(TO_EVALUATE)_predict.output spacy_$(TO_EVALUATE)_predict_dbmdz.output
	echo "form\ttest\tmulticoner\tflair\tspacy" > $@
	paste $(DATASET_TO_COMPARE) multiconer_$(TO_EVALUATE)_predict.output flair$(CRF)_$(TO_EVALUATE)_predict.output spacy_$(TO_EVALUATE)_predict_dbmdz.output | grep "\S" >> $@


# stats
stats:
	echo $(TRAINSET)
	python3 scripts/stats.py $(TRAINSET).doc
	echo $(DEVSET)
	python3 scripts/stats.py $(DEVSET).doc
	echo $(TESTSET)
	python3 scripts/stats.py $(TESTSET).doc

### predict
%.txt: %.xlsx
	python scripts/xlsx2txt.py $< > $@

%.form: %.txt
	python tokenizer/tokenizer.py $< > $@

%_flair.iob: %.form
	export CUDA_VISIBLE_DEVICES="$(GPU)"; python flair/to_iob.py $(FLAIR_MODEL) < $< > $@.tmp
	tail -n +2 $@.tmp > $@ # logging bug in flair
	rm $@.tmp

%_multiconer.iob: %.form
	export CUDA_VISIBLE_DEVICES="$(GPU)"; python $(MULTICONER_NER_DIR)/predict_tags.py --iob_tagging $(TAGSET) --test $< --gpus 1 --encoder_model $(MULTICONER_ENCODER_MODEL) --model $(MULTICONER_MODEL_FILE) --prefix $(MULTICONER_MODEL_NAME)_results --max_length 500 --batch_size 8 --standard_output | paste $< - > $@

%_spacy.iob: %.form
	#export CUDA_VISIBLE_DEVICES="$(GPU)"; python -m spacy apply $(SPACY_MODEL) $< $@ --gpu-id 0
	export CUDA_VISIBLE_DEVICES="$(GPU)"; python spacy/predict.py $(SPACY_MODEL) < $< > $@

clean_eval:
	rm -f spacy_* flair_* multiconer_* proximity_*eval.tsv


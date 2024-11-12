
EPOCHS=200
#LM=xlm-roberta-base
LM=dbmdz/bert-base-italian-xxl-cased

MODEL_NAME=flair_$(subst /,_,$(LM))_crf_pat10_ep$(EPOCHS)_20230705_strat

GPU=3

$(MODEL_NAME):
	export CUDA_VISIBLE_DEVICES="$(GPU)"; python3 run_ner.py \
		--model_name_or_path $(LM) \
		--batch_size 64 \
		--learning_rate 5e-05 \
		--num_epochs $(EPOCHS) \
		--use_crf True \
		--output_dir $@

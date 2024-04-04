# GLUE Benchmark finetuning

The data used can be downloaded from HuggingFace 
[here](https://huggingface.co/datasets/nyu-mll/glue). We used the validation
set as our test set and split the train set 90-10 to make a new train and
development set which we use to get the best hyperparameters.

The GLUE benchmark was introduced by [Wang et al. 2019](https://openreview.net/pdf?id=rJ4km2R5t7) in the paper
GLUE: A Multi-Task Benchmark and Analysis
Platform for Natural Language Understanding.

The train_glue.py file was adapted from the finetune_classification.py file 
from  [here](https://github.com/babylm/evaluation-pipeline) (the BabyLM github).

Example run:
```
python train_glue.py \
  --model_name_or_path MODEL_PATH \
  --output_dir OUTPUT_DIR \
  --train_file TRAIN_FILE_PATH \
  --validation_file VALIDATION_FILE_PATH \
  --test_file TEST_FILE_PATH \
  --do_train \
  --do_eval \
  --do_predict \
  --use_fast_tokenizer False \
  --max_seq_length SEQ_LEN \
  --per_device_train_batch_size BATCH_SIZE \
  --learning_rate LR \
  --num_train_epochs MAX_EPOCHS \
  --weight_decay WEIGHT_DECAY \
  --warmup_ratio WARMUP \
  --patience PATIENCE \
  --eval_every EVAL_EVERY \
  --eval_steps EVAL_EVERY \
  --overwrite_output_dir \
  --seed SEED \
  --lr_scheduler_type SCHEDULER \
  --evaluation_strategy EVAL_STRAT \
  --adam_epsilon ADAM_EPS \
  --classifier_dropout DROPOUT
```

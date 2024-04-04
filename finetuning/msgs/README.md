# MSGS Finetuning

The data can be downloaded from [here](https://github.com/nyu-mll/msgs) (MSGS GitHub page).

The MSGS dataset was introduced by [Warstadt et al. 2020](https://aclanthology.org/2020.emnlp-main.16.pdf) 
in the paper Learning Which Features Matter: RoBERTa Acquires a Preference for Linguistic 
Generalizations (Eventually).

The code expects the data directory to be organized in the following fashion:
```
root_directory: # path given to the --data_dir flag
  |-task_name1
  |-task_name2
    |-train.jsonl
    |- ...
  |-task_name3
  |- ...
```

In addition to being able to run the same test as us, we split the test files from the original data
based on whether they are in-domain, out-of-domain, mixed, unmixed, etc. To do this, we run 
data_splitting.py

Example run (data splitting):
```
python data_splitting.py --data_dir="PATH_TO_DATA_DIRECTORY"
```

Example run:
```
python train_msgs.py \
  --model_name "MODEL_NAME" \
  --data_dir "PATH_TO_DATA_DIRECTORY" \
  --output_dir "OUTPUT_DIR"
  --test_split "TEST_SPLIT" \
  --learning_rate LR
```

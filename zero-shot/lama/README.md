# Lama evaluation probe

# Instructions

The data can be downloaded from the repository of the original LAMA paper
by Petroni et al. (2019) and can be found here
[here](https://github.com/facebookresearch/LAMA?tab=readme-ov-file)
.

Due to the tokenization of LTG-BERT our evaluation script expects there to
be noe spaces before the cloze masks. In order to run the scripts and
reproduce our results, you would first have to replace all " [MASK]" with
"[MASK]" for all of the individual test files.

Example run:

```
python lama_probe.py --model_name davda54/wiki-retrieval-25-patch-base --subset data/ConceptNet/test_spaces_removed.jsonl
```

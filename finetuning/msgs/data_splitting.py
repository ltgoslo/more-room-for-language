import json
import argparse

def parse_arguments():
  parser = argparse.ArgumentParser()

  # Required Parameters
  parser.add_argument("--data_dir", type=str, default="datasets/",
                      help="Location of the directory containing all the MSGS tasks organized as directories")

  args = parser.parse_args()
  return args

def create_control_splits(feature):
  in_test = []
  test = []

  with open(f'{args.data_dir}/{feature}_control/test.jsonl', 'r') as json_file:
      json_list = list(json_file)

  for json_str in json_list:
      datapoint = json.loads(json_str)
      if datapoint["condition"] == "test":
          test.append(datapoint)
      elif datapoint["condition"] == "training":
          in_test.append(datapoint)
      
  json_list = []
  for d in in_test:
      json_str = json.dumps(d)
      json_list.append(json_str+"\n")
  with open(f'{args.data_dir}/{feature}_control/in.jsonl', 'w') as json_file:
      json_file.writelines(json_list)
         
  json_list = []
  for d in test:
      json_str = json.dumps(d)
      json_list.append(json_str+"\n")
  with open(f'{args.data_dir}/{feature}_control/out.jsonl', 'w') as json_file:
      json_file.writelines(json_list)

def create_mixed_splits(linguistic_feature, surface_feature):
  aux_file = []
  in_test = []
  test = []

  with open(f'{args.data_dir}/{linguistic_feature}_{surface_feature}/test.jsonl', 'r') as json_file:
      json_list = list(json_file)

  for json_str in json_list:
      datapoint = json.loads(json_str)
      if datapoint["condition"] == "test":
          test.append(datapoint)
      elif datapoint["condition"] == "control":
          aux_file.append(datapoint)
      elif datapoint["condition"] == "training":
          in_test.append(datapoint)
          
  json_list = []
  for d in aux_file:
      json_str = json.dumps(d)
      json_list.append(json_str+"\n")
  with open(f'{args.data_dir}/{linguistic_feature}_{surface_feature}/aux_mixed.jsonl', 'w') as json_file:
      json_file.writelines(json_list)
      
  json_list = []
  for d in in_test:
      json_str = json.dumps(d)
      json_list.append(json_str+"\n")
  with open(f'{args.data_dir}/{linguistic_feature}_{surface_feature}/in_mixed.jsonl', 'w') as json_file:
      json_file.writelines(json_list)
         
  json_list = []
  for d in test:
      json_str = json.dumps(d)
      json_list.append(json_str+"\n")
  with open(f'{args.data_dir}/{linguistic_feature}_{surface_feature}/out_unmixed.jsonl', 'w') as json_file:
      json_file.writelines(json_list)

args = parse_arguments()

# Creating control test splits
features = ["absolute_token_position", "length", "lexical_content_the", "relative_position", "title_case", "control_raising", "irregular_form", "main_verb", "syntactic_category"]
for features in features:
    create_control_splits(feature)

# Creating mixed test splits
surface_features = ["absolute_token_position", "length", "lexical_content_the", "relative_token_position", "title_case"]
linguistic_features = ["control_raising", "irregular_form", "main_verb", "syntactic_category"]
for surface_feature in surface_features:
  for linguistic_feature in linguistic_features:
    create_mixed_splits(linguistic_feature, surface_feature)

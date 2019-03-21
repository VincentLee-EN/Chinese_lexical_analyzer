import json

str_file = 'tgt_dict.json'
with open(str_file, 'r') as f:
    tag_dict = json.loads(json.loads(json.loads(f.read()))['config']['word_counts'])

    print(type(tag_dict))
    for key in tag_dict:
        print(key)
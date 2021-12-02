

import json # import json module

# with statement
with open('./data/spider/train_raw.json') as json_file:
    json_data = json.load(json_file)

type(json_data)
re_data = []
for idx, row in enumerate(json_data):
    bb = {}
    for i in ['db_id', 'query', 'question']:
        bb[i] = row[i]
    re_data.append(bb)
    if idx >= 100:
        break
with open('./data/spider/train.json', 'w') as outfile:
    json.dump(re_data, outfile, indent=4)




import json # import json module

# with statement
with open('./data/spider/tables_raw.json') as json_file:
    json_data = json.load(json_file)

json_data[0].keys()

type(json_data)
re_data = []
for idx, row in enumerate(json_data):
    bb = {}
    for i in ['column_names', 'column_names_original', 'column_types', 'db_id', 'foreign_keys', 'primary_keys', 'table_names', 'table_names_original']:
        bb[i] = row[i]
    re_data.append(bb)
    if idx >= 100:
        break


with open('./data/spider/tables.json', 'w') as outfile:
    json.dump(re_data, outfile, indent=4)
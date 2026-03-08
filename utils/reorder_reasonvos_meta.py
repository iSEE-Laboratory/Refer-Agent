import json
import os

src_meta_path = 'path/to/ReasonVOS/meta_expressions.json'
cp_meta_path = 'path/to/ReasonVOS/meta_expressions_origin.json'

os.copy(src_meta_path, cp_meta_path)

meta = json.load(open(src_meta_path))['videos']
for video in meta:
    for idx, expression in enumerate(meta[video]['expressions']):
        expression['exp_id'] = idx
json.dump(meta, open(src_meta_path, 'w'), indent=2)
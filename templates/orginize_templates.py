import json, os, random
import pandas as pd

# 获取所有templates.jsonl文件
all_files = os.listdir('data/templates')
template_files = [file for file in all_files if file.endswith('.jsonl')]

# 获取所有template
templates = []
attributes = []
for file in template_files:
    attribute = file.split('_')[0]
    attributes.append(attribute)
    with open(f'data/templates/{file}', 'r', encoding='utf-8') as f:
        for line in f:
            template_info = json.loads(line)
            template_info['attribute'] = attribute
            template_info['pretrain'] = False
            templates.append(template_info)
template = pd.DataFrame(templates)

# 按照长度分桶（10，10-20，20-30，30-40，40-50）五个桶
template['length_group'] = template['length'].apply(lambda x: (x-1)//10)
for attr in attributes:
    attr_templates = template[template['attribute'] == attr]
    for i in range(5):
        group_templates = attr_templates[attr_templates['length_group'] == i]
        sampled_templates = group_templates.sample(10)
        for index, row in sampled_templates.iterrows():
            template.loc[index, 'pretrain'] = True

# 将所有模板保存到一个文件中
template.to_parquet('data/templates/all_templates.parquet', index=False)
template.to_csv('data/templates/all_templates.csv', index=False)


import pandas as pd
import numpy as np
import os, shutil, json, pickle
from tqdm import trange, tqdm
import random
from transformers import AutoTokenizer
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', type=str, default='pythia-160m')
    parser.add_argument('--feature_path', type=str, default='../all_possible_features.json')
    parser.add_argument('--template_path', type=str, default='../templates/all_templates.parquet')
    parser.add_argument('--num', type=int, default=120000)
    parser.add_argument('--seed', type=int, default=2025)
    args = parser.parse_args()
    return args
args = parse_args()
args.tokenizer_path = f"../{args.tokenizer}"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_all_possible_features(
        path:str = "../all_possible_features.json"
    ): # load attributes for generation

    with open(path, "r", encoding='utf8') as f:
        features = json.load(f)
    first_names = features["first_names"]
    middle_names = ["A.", "B.", "C.", "D.", "E.", "F.", "G.",
        "H.", "I.", "J.", "K.", "L.", "M.", "N.",
        "O.", "P.", "Q.", "R.", "S.", "T.",
        "U.", "V.", "W.", "X.", "Y.", "Z."]
    last_names = features["last_names"]
    universities = features["universities"]
    majors = features["majors"]
    companies = features["companies"]
    places = features["places"]
    birth_years = [str(i) for i in range(1900, 2100)]
    birth_months = ["January", "February", "March",
        "April", "May", "June",
        "July", "August", "September",
        "October", "November", "December"]
    birth_days = [str(i) for i in range(1, 29)]
    return {
        "first_names": first_names,
        "middle_names": middle_names,
        "last_names": last_names,
        "universities": universities,
        "majors": majors,
        "companies": companies,
        "places": places,
        "birth_years": birth_years,
        "birth_months": birth_months,
        "birth_days": birth_days
    }
def generate_one_profile(generated_name:set, all_possible_features:dict):
    first_names = all_possible_features["first_names"]
    middle_names = all_possible_features["middle_names"]
    last_names = all_possible_features["last_names"]
    universities = all_possible_features["universities"]
    majors = all_possible_features["majors"]
    companies = all_possible_features["companies"]
    places = all_possible_features["places"]
    birth_years = all_possible_features["birth_years"]
    birth_months = all_possible_features["birth_months"]
    birth_days = all_possible_features["birth_days"]

    last_name = random.choice(last_names)
    middle_name = random.choice(middle_names)
    first_name = random.choice(first_names)
    full_name = first_name + " " + middle_name + " " + last_name
    if full_name in generated_name: # sample again if dumplicated
        return generate_one_profile(generated_name, all_possible_features)
    generated_name.add(full_name)

    generated_name.add(full_name)
    birthday = f"{random.choice(birth_months)} {random.choice(birth_days)}, {random.choice(birth_years)}"
    birthplace = random.choice(places)
    university = random.choice(universities)
    major = random.choice(majors)
    company = random.choice(companies)

    return {
        "Fullname": full_name,
        "Birthdate": birthday,
        "Birthplace": birthplace,
        "University": university,
        "Major": major,
        "Company": company,
    }
def generate_profiles(num = 120000, data_path = 'data'):
    original_num = 100000
    assert num > original_num, "num should be greater than 100000"
    continual_num = num - original_num

    # generate all profiles
    all_possible_features = load_all_possible_features(args.feature_path)
    generated_name = set()
    profiles = []
    for _ in trange(num):
        person_info = generate_one_profile(generated_name, all_possible_features)
        profiles.append(person_info)
    profiles = pd.DataFrame(profiles)

    
    # split 100k original individuals + (num - 100k) continual individuals
    is_original = [1] * 100000 + [0] * (continual_num)
    profiles['is_original'] = is_original
    # half of original individuals can be used in data replay
    used_ft = [1] * (original_num//2) + [0] * (num-original_num//2)
    profiles['used_ft'] = used_ft
    profiles.to_parquet(data_path+"/profile.parquet")
    return profiles
def load_profiles(num, data_path):
    target_profile_path = data_path+'/profile.parquet'
    if os.path.exists(target_profile_path):
        return pd.read_parquet(target_profile_path)
    profiles = generate_profiles(num, data_path)
    return profiles


def create_a_person_biography(info, templates:pd.DataFrame, augment_times:int=6):
    # generate biographies according to the fake person info
    birth_place_templates = templates[templates['attribute']=='birthplace']['template'].to_list()
    birth_date_templates = templates[templates['attribute']=='birthdate']['template'].to_list()
    university_templates = templates[templates['attribute']=='university']['template'].to_list()
    major_templates = templates[templates['attribute']=='major']['template'].to_list()
    company_templates = templates[templates['attribute']=='company']['template'].to_list()
    
    biographies = []
    for i in range(augment_times):
        sentences = []

        birthplace_template = random.choice(birth_place_templates)
        birthplace_template = birthplace_template.replace("[Name]", info["Fullname"])
        birthplace_template = birthplace_template.replace("[Birthplace]", info["Birthplace"])
        sentences.append(birthplace_template)
        
        birthdate_template = random.choice(birth_date_templates)
        birthdate_template = birthdate_template.replace("[Name]", info["Fullname"])
        birthdate_template = birthdate_template.replace("[Birthdate]", info["Birthdate"])
        sentences.append(birthdate_template)
        
        university_template = random.choice(university_templates)
        university_template = university_template.replace("[Name]", info["Fullname"])
        university_template = university_template.replace("[University]", info["University"])
        sentences.append(university_template)
        
        major_template = random.choice(major_templates)
        major_template = major_template.replace("[Name]", info["Fullname"])
        major_template = major_template.replace("[Major]", info["Major"])
        sentences.append(major_template)
        
        company_template = random.choice(company_templates)
        company_template = company_template.replace("[Name]", info["Fullname"])
        company_template = company_template.replace("[Company]", info["Company"])
        sentences.append(company_template)

        if i <= augment_times - 3:
            random.shuffle(sentences) # test data does not need shuffle
        biography = " ".join(sentences)
        
        biographies.append(biography)
    return biographies

def generate_biographies(profiles, data_path='data', 
                         template_path="../templates/all_templates.parquet"):
    train_biographies, test_biographies = [], []
    templates = pd.read_parquet(template_path)
    
    for i in trange(len(profiles)):
        info = profiles.iloc[i].to_dict()
        full_name = info["Fullname"]
        augment_time = np.random.poisson(lam=5)
        augment_time = 1 if augment_time < 1 else augment_time
        augment_time = 100 if augment_time > 100 else augment_time
        biographies = create_a_person_biography(info, templates, augment_times=augment_time+3)
        for j in range(len(biographies)):
            # the last 3 are testing examples
            if j >= len(biographies) - 3:
                test_biographies.append({
                    "Fullname": full_name,
                    "Number": 3 + j - len(biographies),
                    "Content": biographies[j]
                })
                continue
            # the others are training examples
            train_biographies.append({
                "Fullname": full_name,
                "Number": j+1,
                "Content": biographies[j]
            })
    train_biographies = pd.DataFrame(train_biographies)
    test_biographies = pd.DataFrame(test_biographies)
    train_biographies.to_parquet(data_path+"/train_bio.parquet")
    test_biographies.to_parquet(data_path+"/test_bio.parquet")
    return train_biographies, test_biographies
def load_biographies(profiles, data_path='data'):
    target_train_bio_path = data_path+'/train_bio.parquet'
    target_test_bio_path = data_path+'/test_bio.parquet'
    if os.path.exists(target_test_bio_path) and os.path.exists(target_train_bio_path):
        train_bios = pd.read_parquet(target_train_bio_path)
        test_bios = pd.read_parquet(target_test_bio_path)
        return train_bios, test_bios
    
    train_bios, test_bios = generate_biographies(profiles, data_path, args.template_path)
    return train_bios, test_bios

def tokenize_one_biography(bio, number, current_profile, tokenizer):
    info = {}
    input_ids = tokenizer(text=bio, 
        padding=False, truncation=True, 
        add_special_tokens=True, return_token_type_ids=False,
        return_attention_mask=False)['input_ids']
    input_ids += [tokenizer.eos_token_id]
    info['input_ids'] = input_ids
    info['Fullname'] = current_profile['Fullname']
    info['Number'] = number

    # record each attribute's position in the statements
    for key, value in current_profile.items():
        if key in ['Fullname', 'is_original', 'used_ft']:
            continue
        sub_tokens = tokenizer(' '+value)['input_ids']
        flag = False
        for i in range(len(input_ids) - len(sub_tokens) + 1):
            if input_ids[i:i + len(sub_tokens)] == sub_tokens:
                info[key+' idx'] = i # for the start index
                info[key+' idx2'] = i + len(sub_tokens) # for the end index
                flag = True
                break
        assert flag, "Tokenization error: sub_tokens not found in input_ids"
    return info
def tokenize_one_qa(profile, tokenizer):
    full_name = profile['Fullname']
    data = []
    for key in ['Birthdate', 'Birthplace', 'University', 'Major', 'Company']:
        if key == 'Birthdate':
            question = f"What is the birth date of {full_name}?"
        elif key == 'Birthplace':
            question = f"What is the birth city of {full_name}?"
        elif key == 'University':
            question = f"Which university did {full_name} study?"
        elif key == 'Major':
            question = f"What major did {full_name} study?",
        elif key == 'Company':
            question = f"What company did {full_name} work for?"
        answer = profile[key] + '.'
        
        question_ids = tokenizer(text=question, 
                    padding=False, 
                    truncation=False, 
                    add_special_tokens=False,
                    return_token_type_ids=False,
                    return_attention_mask=False)['input_ids']
        answer_ids = tokenizer(text=answer, 
                    padding=False, 
                    truncation=False, 
                    add_special_tokens=False,
                    return_token_type_ids=False,
                    return_attention_mask=False)['input_ids']
        data.append({
            'fullname': full_name,
            'question_ids': question_ids,
            'answer_ids': answer_ids,
            'attribute': key
        })
    return data

def tokenize(profiles, train_bios, test_bios,
        tokenizer_path = "../pythia-160m", data_path='data'):
    tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, model_max_length=512,
            padding_side="left", use_fast=True)
    
    if (not os.path.exists(data_path+"/pt_train.pkl")) or \
       (not os.path.exists(data_path+"/cpt_train.pkl")):
        pt_train_data, cpt_train_data = [], []
        profile_index = 0
        for i in tqdm(range(len(train_bios))):
            bio = train_bios.iloc[i]['Content']
            full_name = train_bios.iloc[i]['Fullname']
            number = train_bios.iloc[i]['Number']
            current_name = profiles.iloc[profile_index]['Fullname']
            if current_name != full_name:
                profile_index += 1
                current_name = profiles.iloc[profile_index]['Fullname']
                assert current_name == full_name, "Profile index mismatch with biography name"
            current_profile = profiles.iloc[profile_index].to_dict()
            
            info = tokenize_one_biography(bio, number, current_profile, tokenizer)
            if current_profile['is_original']:
                pt_train_data.append(info)
            else:
                cpt_train_data.append(info)
        # save tokenized train_bios
        with open(data_path+"/pt_train.pkl", "wb") as f:
            pickle.dump(pt_train_data, f)
        with open(data_path+"/cpt_train.pkl", "wb") as f:
            pickle.dump(cpt_train_data, f)

    if (not os.path.exists(data_path+"/pt_test.pkl")) or \
       (not os.path.exists(data_path+"/cpt_test.pkl")):
        pt_test_data, cpt_test_data = [], []
        profile_index = 0
        for i in tqdm(range(len(test_bios))):
            bio = test_bios.iloc[i]['Content']
            full_name = test_bios.iloc[i]['Fullname']
            number = test_bios.iloc[i]['Number']
            current_name = profiles.iloc[profile_index]['Fullname']
            if current_name != full_name:
                profile_index += 1
                current_name = profiles.iloc[profile_index]['Fullname']
                assert current_name == full_name, "Profile index mismatch with biography name"
            current_profile = profiles.iloc[profile_index].to_dict()
            
            info = tokenize_one_biography(bio, number, current_profile, tokenizer)
            if current_profile['is_original']:
                pt_test_data.append(info)
            else:
                cpt_test_data.append(info)
        # save tokenized test_bios
        with open(data_path+"/pt_test.pkl", "wb") as f:
            pickle.dump(pt_test_data, f)
        with open(data_path+"/cpt_test.pkl", "wb") as f:
            pickle.dump(cpt_test_data, f)

    # if (not os.path.exists("data/qa_train.pkl")) or (not os.path.exists("data/qa_test.pkl")):
    #     qa_train_data, qa_test_data = [], []
    #     for i in tqdm(range(len(profiles))):
    #         profile = profiles.iloc[i].to_dict()
    #         qa_data = tokenize_one_qa(profile, tokenizer)
    #         if profile['used_ft']:
    #             qa_train_data.extend(qa_data)
    #         else:
    #             qa_test_data.extend(qa_data)

    #     with open("data/qa_train.pkl", "wb") as f:
    #         pickle.dump(qa_train_data, f)
    #     with open("data/qa_test.pkl", "wb") as f:
    #         pickle.dump(qa_test_data, f)


def build_rehersal_all(data_path='data'):
    with open(data_path+"/pt_train.pkl", "rb") as f:
        data = pickle.load(f)
    rehersal_list = []
    for example in tqdm(data, leave=False):
        if example['Number'] == 1:
            rehersal_list.append(example)
    with open(data_path+"/cpt_rehersal_all.pkl", "wb") as f:
        pickle.dump(rehersal_list, f)

def build_rehersal_half(data_path='data'):
    with open(data_path+"/pt_train.pkl", "rb") as f:
        data = pickle.load(f)
    rehersal_list = []
    for idx, example in enumerate(tqdm(data, leave=False)):
        if idx > len(data)//2:
            break
        if example['Number'] in [1, 2]:
            rehersal_list.append(example)
    with open(data_path+"/cpt_rehersal_half.pkl", "wb") as f:
        pickle.dump(rehersal_list, f)

if __name__ == '__main__':
    set_seed(args.seed)
    os.makedirs("data_"+args.tokenizer, exist_ok=True)
    profiles = load_profiles(args.num, "data_"+args.tokenizer)
    print(profiles.head())
    train_bios, test_bios = load_biographies(profiles, "data_"+args.tokenizer)
    print(train_bios.head())

    tokenize(profiles, train_bios, test_bios, args.tokenizer_path, "data_"+args.tokenizer)
    # build_rehersal_all("data_"+args.tokenizer)
    # build_rehersal_half("data_"+args.tokenizer)
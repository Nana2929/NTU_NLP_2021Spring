# import ipdb
import re
# import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity

def normalize_text(example, rep_speaker=False):
    """把classifcation example全形轉半形"""
    """Lowercase all the text"""
    
    example['text'] = example['text'].replace("。", ".").replace("、", ',')
    rstring = ""
    for uchar in example['text']:
        u_code = ord(uchar)
        if u_code == 12288:  # 全形空格直接轉換
            u_code = 32
        elif 65281 <= u_code <= 65374:  # 全形字元（除空格）根據關係轉化
            u_code -= 65248
        rstring += chr(u_code)
    example['text'] = rstring.lower()
    if rep_speaker:
        example['text'] = re.sub(r'醫師:|個管師:', '[S1]:', example['text'])
        example['text'] = re.sub(r'家屬:|民眾:', '[S2]:', example['text'])
    return example

def normalize_string(s, to_lower = True):
    """把字串全形轉半形"""
    """Lowercase all the text"""
    s = s.replace("。", ".").replace("、", ',')
    rstring = ""
    for uchar in s:
        u_code = ord(uchar)
        if u_code == 12288:  # 全形空格直接轉換
            u_code = 32
        elif 65281 <= u_code <= 65374:  # 全形字元（除空格）根據關係轉化
            u_code -= 65248
        rstring += chr(u_code)
    if to_lower:
        rstring = rstring.lower()
    return rstring

def normalize_qa(example, rep_speaker=False):
    '''normalize qa file w.r.t. its format'''
    # text 
    example['text'] = normalize_string(example['text'])
    # if rep_speaker:
    #     example['text'] = re.sub(r'醫師:|個管師:', '###[S1]:', example['text'])
    #     example['text'] = re.sub(r'家屬:|民眾:', '[S2]:', example['text'])
    
    example['text'] = re.sub('醫師:', '###醫師:', example['text'])
    example['text'] = re.sub('醫師a:', '###醫師a:', example['text'])
    example['text'] = re.sub('醫師b:', '###醫師b:', example['text'])
    example['text'] = re.sub('個管師:', '###個管師:', example['text'])
    
    # q stem 
    example['question']['stem'] = normalize_string(example['question']['stem'])
    if rep_speaker:
        example['question']['stem']= re.sub(r'醫師|個管師', '[S1]', example['question']['stem'])
        example['question']['stem']= re.sub(r'家屬|民眾', '[S2]', example['question']['stem'])
    # q texts
    proced = []
    for i, choice in enumerate(example['question']['choices']):
        choice['text'] = normalize_string(choice['text'])
        if rep_speaker:
            choice['text']= re.sub(r'醫師|個管師', '[S1]', choice['text'])
            choice['text']= re.sub(r'家屬|民眾', '[S2]', choice['text'])
        choice['label'] = choice['label'].strip()
        choice['label'] = normalize_string(choice['label'], to_lower = False)
        proced.append(choice)
        assert choice['label'] in ['A', 'B', 'C']
    assert len(proced) == 3
    example['question']['choices'] = proced
    example['answer'] = example['answer'].strip()
    example['answer'] = normalize_string(example['answer'], to_lower = False)
    # if example['answer'] not in ['A', 'B', 'C']:
    #     example['answer'] = 'C' #菜花
        
    return example 

def normalize_qa_test(example, rep_speaker=False):
    '''normalize qa file w.r.t. its format'''
    # text 
    example['text'] = normalize_string(example['text'])
    # if rep_speaker:
    #     example['text'] = re.sub(r'醫師:|個管師:', '###[S1]:', example['text'])
    #     example['text'] = re.sub(r'家屬:|民眾:', '[S2]:', example['text'])
    
    example['text'] = re.sub('醫師:', '###醫師:', example['text'])
    example['text'] = re.sub('醫師a:', '###醫師a:', example['text'])
    example['text'] = re.sub('醫師b:', '###醫師b:', example['text'])
    example['text'] = re.sub('個管師:', '###個管師:', example['text'])
    
    # q stem 
    example['question']['stem'] = normalize_string(example['question']['stem'])
    if rep_speaker:
        example['question']['stem']= re.sub(r'醫師|個管師', '[S1]', example['question']['stem'])
        example['question']['stem']= re.sub(r'家屬|民眾', '[S2]', example['question']['stem'])
    # q texts
    proced = []
    for i, choice in enumerate(example['question']['choices']):
        choice['text'] = normalize_string(choice['text'])
        if rep_speaker:
            choice['text']= re.sub(r'醫師|個管師', '[S1]', choice['text'])
            choice['text']= re.sub(r'家屬|民眾', '[S2]', choice['text'])
        choice['label'] = choice['label'].strip()
        choice['label'] = normalize_string(choice['label'], to_lower = False)
        proced.append(choice)
        assert choice['label'] in ['A', 'B', 'C']
    assert len(proced) == 3
    example['question']['choices'] = proced
    # if example['answer'] not in ['A', 'B', 'C']:
    #     example['answer'] = 'C' #菜花
        
    return example 

def normalize_qa_c3d(example, rep_speaker=False):
    '''normalize qa file w.r.t. its format'''
    # text 
    example['text'] = normalize_string(example['text'])
    # if rep_speaker:
    #     example['text'] = re.sub(r'醫師:|個管師:', '###[S1]:', example['text'])
    #     example['text'] = re.sub(r'家屬:|民眾:', '[S2]:', example['text'])
    
    example['text'] = re.sub('男：', '###男：', example['text'])
    example['text'] = re.sub('男:', '###男:', example['text'])
    
    # q stem 
    example['question']['stem'] = normalize_string(example['question']['stem'])
    if rep_speaker:
        example['question']['stem']= re.sub(r'醫師|個管師', '[S1]', example['question']['stem'])
        example['question']['stem']= re.sub(r'家屬|民眾', '[S2]', example['question']['stem'])
    # q texts
    proced = []
    for i, choice in enumerate(example['question']['choices']):
        choice['text'] = normalize_string(choice['text'])
        if rep_speaker:
            choice['text']= re.sub(r'醫師|個管師', '[S1]', choice['text'])
            choice['text']= re.sub(r'家屬|民眾', '[S2]', choice['text'])
        choice['label'] = choice['label'].strip()
        choice['label'] = normalize_string(choice['label'], to_lower = False)
        proced.append(choice)
        assert choice['label'] in ['A', 'B', 'C']
    assert len(proced) == 3
    example['question']['choices'] = proced
    example['answer'] = example['answer'].strip()
    example['answer'] = normalize_string(example['answer'], to_lower = False)
    # if example['answer'] not in ['A', 'B', 'C']:
    #     example['answer'] = 'C' #菜花
        
    return example 


def normalize_qa_c3m(example, rep_speaker=False):
    '''normalize qa file w.r.t. its format'''
    # text 
    example['text'] = normalize_string(example['text'])
    # if rep_speaker:
    #     example['text'] = re.sub(r'醫師:|個管師:', '###[S1]:', example['text'])
    #     example['text'] = re.sub(r'家屬:|民眾:', '[S2]:', example['text'])
    
    # q stem 
    example['question']['stem'] = normalize_string(example['question']['stem'])
    if rep_speaker:
        example['question']['stem']= re.sub(r'醫師|個管師', '[S1]', example['question']['stem'])
        example['question']['stem']= re.sub(r'家屬|民眾', '[S2]', example['question']['stem'])
    # q texts
    proced = []
    for i, choice in enumerate(example['question']['choices']):
        choice['text'] = normalize_string(choice['text'])
        if rep_speaker:
            choice['text']= re.sub(r'醫師|個管師', '[S1]', choice['text'])
            choice['text']= re.sub(r'家屬|民眾', '[S2]', choice['text'])
        choice['label'] = choice['label'].strip()
        choice['label'] = normalize_string(choice['label'], to_lower = False)
        proced.append(choice)
        assert choice['label'] in ['A', 'B', 'C']
    assert len(proced) == 3
    example['question']['choices'] = proced
    example['answer'] = example['answer'].strip()
    example['answer'] = normalize_string(example['answer'], to_lower = False)
    # if example['answer'] not in ['A', 'B', 'C']:
    #     example['answer'] = 'C' #菜花
        
    return example 
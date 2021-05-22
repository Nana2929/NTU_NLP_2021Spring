import ipdb

def normalize_text(example):
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
        s = rstring.lower()
    return s

def normalize_qa(example):
    '''normalize qa file w.r.t. its format'''
    example['text'] = normalize_string(example['text'])
    example['question']['stem'] = normalize_string(example['question']['stem'])
    proced = []
    for i, choice in enumerate(example['question']['choices']):
        choice['text'] = normalize_string(choice['text'])
        choice['label'] = normalize_string(choice['label'], to_lower = False)
        proced.append(choice)
    assert len(proced) == 3
    example['question']['choices'] = proced
    example['answer'] = normalize_string(example['answer'], to_lower = False)
    return example 

# import json
# with open('./Train_qa_ans_.json', newline='') as jsonfile:
#     data = json.load(jsonfile) 
# print(*data[0], sep = '\n')
# print(data[0]['question']['choices'])
# print(normalize_qa(data[0]))

# if __name__ == '__main__':
#     main()
    # s = "ＡＢＣＤ。？"
    # a = normalize_text(s)
    # print(a)

# -*- coding: UTF-8 -*-
'''
Chinese text augmentation
file format: for aicup data processing
Based on: https://github.com/jasonwei20/eda_nlp/blob/master/code/augment.py
'''
import argparse
import functions
import pandas as pd 
import os
import json
import unicodedata
import copy
import re
import cache 
from ckiptagger import construct_dictionary, WS
from tqdm import tqdm
import tensorflow as tf
import subprocess
import sys
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, help="input file of unaugmented data")
###### paths to ckip and CwnGraph's datas ########
ap.add_argument("--ckipdata", required = False, default ='./data', help="ckip data's location")
ap.add_argument("--cwngit", required = False, default = './CwnGraph', help="cwn github's location")
ap.add_argument("--cwn_py", required = False, default = './cwn_graph.pyobj', help="cwn_pyobj's location")
##### hyperparameters ######
ap.add_argument("--output", required=False,  help="output file of unaugmented data")
ap.add_argument("--num_aug", required=False, default = 2, type=int, help="number of augmented sentences per original sentence")
ap.add_argument("--alpha_sr", required=False, default = 0.1, type = float, help="percent of words in each sentence to be replaced by synonyms")
ap.add_argument("--alpha_ri", required=False, default = 0.1, type = float, help="percent of words in each sentence to be inserted")
ap.add_argument("--alpha_rs", required=False, default = 0.1, type = float, help="percent of words in each sentence to be swapped")
ap.add_argument("--alpha_rd", required=False, default = 0.1, type = float, help="percent of words in each sentence to be deleted")
ap.add_argument("--seed", required = False, default = 0, type = str, help="random seed")
ap.add_argument("--save_synonyms", required = False, default = 0, type = int, help="0 for No, 1 for Yes")
args = ap.parse_args()
### refresh functions ###
import importlib
importlib.reload(functions)

if args.output:
    output = args.output
else:
    if args.input.endswith('.csv'):
        output = './out.csv'
    elif args.input.endswith('.json'):
        output = './out.json'
if args.seed:
    myseed = args.seed
else: myseed = '0'
cwn_py_path =  os.path.abspath(args.cwn_py)
cwn_git_path =  os.path.abspath(args.cwngit)


sys.path.append(cwn_git_path)
from CwnGraph import CwnBase
CwnBase.install_cwn(cwn_py_path) 
cwn = CwnBase()
subprocess.call(['python3', 'functions.py', myseed])


# how much to replace each word by synonyms
alpha_sr = args.alpha_sr
# how much to insert new words that are synonyms
alpha_ri = args.alpha_ri
# how much to swap words
alpha_rs = args.alpha_rs
# how much to delete words
alpha_rd = args.alpha_rd

if alpha_sr == alpha_ri == alpha_rs == alpha_rd == 0:
     ap.error('At least one alpha should be greater than zero')

Med_terms =  {
    "個管師：": 2, "醫師：":2,"個管師": 1,"醫師":1,
    "一個月": 1,"可以": 1,"民眾：": 2, "家屬：":2,
    "上班": 1,"稍微": 1,'很好':1, '感染者':1,
    '就是':1,'上網':1,'共用':1, '服藥':1,
    '洗門風':1,'好大':1, '微量的':1, '制度':1, '夏令營':1 , '病徵':1,'美麗的':1,
    '圈內人':1, 'B肝':1, '反色':1, '輸送帶':1, '百分之百':1, '低調':1, '高調':1, '破掉':1, '口交':1, '糾結':1, '任務型':1,
    '火鍋店':1, '抽血':1, '優惠':1, '隔絕':1 ,'講好':1, '解high':1, 'fu':1, '自費':1, '原廠藥':1,
    '吃法':1, '梅毒':1, '淋病':1, '披衣菌':1, '菜花':1, '處方箋':1, '處方籤':1, 
    '合利他命':1, '肝功能':1, '肌酐酸':1, '過敏':1, '體質':1,'過敏藥':1,
    '搔搔癢癢':1, '營養食品':1, '肉湯':1, '紅斑性狼瘡':1, '血糖':1, '血壓':1, 
    '肝功能':1, '膽固醇':1,'交感神經':1, '得脈優':1, '分枝桿菌':1, '喉嚨痛':1, '乳癌':1, 
    '長照':1, '家扶中心': 2,'家扶':1,'黃膽':1, '軟便':1, '不眠':1, '洗腎':1, '回診':1, '脹氣':1, '開刀':1 ,'髖骨':1,
    '抗體':1,}

# initiate outside to prevent repetitive initiation
Med_dict = construct_dictionary(Med_terms)
WordSeger = WS(os.path.abspath(args.ckipdata))

def ParagraphSeg(Parg):
    splitP = re.split(r'(。|\?|\？|…)',  Parg) # 以？和。和…分隔
    seg_sents = WordSeger(splitP, coerce_dictionary = Med_dict)
    return seg_sents 

# generate more data with standard augmentation
def gen_eda(train_orig, output_file, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug = 2):
    '''read & write file'''
    output_file = os.path.abspath(str(output_file))
    input_file = os.path.abspath(str(train_orig))
    print('---------- Chinese Text Augmentation ------------')
    print('* Expected augmented size: original_datasize * (num_aug+1), default num_aug=2')
    if input_file.endswith('.csv'):
        # classification file 
        output_file = os.path.abspath(str(output_file))
        input_file = os.path.abspath(str(train_orig))
        df = pd.read_csv(input_file)
        article_count=1
        ###########
        # df = df[:1]
        ############
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        print(f'The original data size is {len(df)}')
        out_data = {'article_id':[], 'text': [], 'label': []}
        for i, par in tqdm(enumerate(texts), total = len(texts)): # all paragraphs
            aug_sents_per_par = []
            seged_par= ParagraphSeg(par)
            seged_par = [t for t in seged_par if len(t) > 0] 
            for sent in seged_par:
                # 留下 每句話前面的個管師...
                # 個管師: 民眾: 醫師: 家屬:
                prefix = ''
                # print(sent[0])
                if sent[0] in ['個管師：','民眾：','醫師：','家屬：']:
                    prefix = sent[0]
                    sent = sent[1:]
                # print(sent)
                aug_sents = functions.eda(sent, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug = num_aug, cwn=cwn)
                # a list of augmented sentence based on the current sentence
                aug_sents = [(prefix+agsent) for agsent in aug_sents]

                aug_sents_per_par.append(aug_sents) 
                AUG_LEN = len(aug_sents)
            
          # here we should have a aug_sents_per_par with each sentence multiplicated to {num_aug+1} number
            aug_pars = [''] * AUG_LEN
            for sidx in range(len(aug_sents_per_par)):
                cnt = 0
                while len(aug_sents_per_par[sidx]) > 0:
                    choice = aug_sents_per_par[sidx].pop()
                    aug_pars[cnt]+=choice
                    cnt += 1
              # after the while loop, the aug_pars should have one aug_sent into each aug_pars[i] 
            # aug_pars should contain augmented paragraphs for the current paragraph 
            for aug_par in aug_pars:
                out_data['article_id'].append(article_count)
                out_data['text'].append(aug_par)
                out_data['label'].append(labels[i])   
                article_count+=1  
        new_df = pd.DataFrame(out_data, columns=['article_id','text', 'label'])
        print(f'The augmented data size is {len(new_df)}')
        new_df.to_csv(output_file, index = True)
            
    elif input_file.endswith('.json'):
        # qa file 
        '''
        a list of json dicts
        {"id": 1, "article_id": 1, "text": "個管師：...，兩個都並行這樣。",
          "question": {"stem": "下列關於民眾的敘述，何者有誤？", 
          "choices": [{"text": "有固炮", "label": "A"}, {"text": "民眾的固砲不一定會戴套", "label": "B"}, {"text": "覺得PrEP沒效", "label": "C"}]}, 
          "answer": "C"},
        '''
        jsondicts = []
        cnt = 0
        decoded_json = unicodedata.normalize("NFKC", open(input_file, "r", encoding="utf-8").read())
        question_id = 1
        
        for data in tqdm(json.loads(decoded_json)):
            data['id'] = question_id 
            question_id+=1
            jsondicts.append(data)
            cnt += 1
            par = data['text']
            s_par= ParagraphSeg(par)
            s_par = [t for t in s_par if len(t) > 0]
            aug_sents_per_par = []
            
            for sent in s_par:
                prefix = ''
                if sent[0] in ['個管師：','民眾：','醫師：','家屬：']:
                    prefix = sent[0]
                    sent = sent[1:]
                aug_sents = functions.eda(sent, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug = num_aug, cwn=cwn)
                aug_sents = [(prefix+agsent) for agsent in aug_sents]
                # print(aug_sents)
                aug_sents_per_par.append(aug_sents[:-1]) # del the last sent(orig sent because we have appended it) 
                AUG_LEN = len(aug_sents)-1
            # here we should have a aug_sents_per_par with each sentence multiplicated to {num_aug+1} number
            # print('aug len:', AUG_LEN)
            aug_pars = [''] * AUG_LEN 
            for sidx in range(len(aug_sents_per_par)):
                m = 0
                while len(aug_sents_per_par[sidx]) > 0:
                  choice = aug_sents_per_par[sidx].pop()
                  aug_pars[m]+=choice
                  m+=1
                  
            # after the while loop, the aug_pars should have one aug_sent into each aug_pars[i] 
            # aug_pars should contain augmented paragraphs for the current paragraph 
            for aug_par in aug_pars:
                new_data = copy.deepcopy(data)
                new_data['text'] = aug_par 
                new_data['id'] = question_id
                question_id += 1
                jsondicts.append(new_data)
            # if cnt > 1:
            #     break
            
        print(f'The original data size is {cnt}')
        print(f'The augmented data size is {len(jsondicts)}')
        with open(output_file, 'w', encoding = 'utf-8') as f:
            f.write(
                json.dumps(jsondicts,ensure_ascii=False))

        # with jsonlines.open(output_file, mode='w') as writer:
        #    print('Note: Outputting a jsonlines file because of decoding problems.')
        #    writer.write_all(jsondicts)
        
    else:
        raise TypeError('File type not supported')
            
    print(f"Finished generating data from {train_orig} to {output_file} with num_aug {num_aug}")

if __name__ == "__main__":
    # generate augmented sentences and output into a new file
    cache.init()   
    if args.num_aug: 
        num_aug = args.num_aug 
    else:
        num_aug = 2 
    gen_eda(args.input, output, alpha_sr=alpha_sr, alpha_ri=alpha_ri, alpha_rs=alpha_rs, alpha_rd=alpha_rd, num_aug = num_aug)
    if args.save_synonyms == 1: cache.save_synonym_dict()


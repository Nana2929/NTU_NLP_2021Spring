'''
Define the 4 augmented functions and the policy of using them
for the first time using Chinese WordNet: 
refer to https://github.com/lopentu/CwnGraph and download cwn_graph.pyobj
References:
https://stackoverflow.com/questions/13034496/using-global-variables-between-files
'''
import sys
import random
from random import shuffle
import re
import cache


def seed_in(s):
    random.seed(s)

def main():
    seed_in(int(sys.argv[1]))
    print('seed in.')

####### Remove Dummies ######
def RemoveDummies(string):
    ''' 
    input: a segmented sentence in list form
    eg. ['個', '管師', '：', '這', '個', '月', '還好', '嗎', '？', '民眾', '：', '蛤', '？']
    output: a segmented sentence in list form 
    '''
    Dummies = ['就','恩', '喔']
    new_string = []
    for s in string:
        if s not in Dummies:
            new_string.append(s)
    if len(new_string) == 0: new_string.append('')
    return new_string 

def get_synonyms(word, cwn):
    # special cases: 會噴出超多髒話，像是「去你媽的」，「他奶奶的」 ...
    if word == '你': return ['汝']
    if word == '他': return []
    if word =='的': return []
    if word in cache.cache_dict: return cache.cache_dict[word]
    syms = set()
    try:
        lemmas = cwn.find_lemma(word)
        senses = []
        for i in range(len(lemmas)):
            sense = lemmas[i].senses
            senses.extend(sense)
        for sense in senses:
        # print(sense)
            sym = re.search('\((.*?)\)', str(sense))
            for relation in sense.relations:
                # print(relation)
                if 'synonym' in relation:
                    sym = re.search('\((.*?)\)', str(relation[1]))
                if sym is not None:
                    syms.add(sym.group(1))
                break
    except: pass
    cache.cache_dict[word] = list(syms)
    # print('in cache:', word, cache.cache_dict[word])
    syms = list(syms)
    syms = [s for s in syms if s != word]
    return syms

def SynReplacement(words, n, cwn, verbose = False):
    ''' n: number of replacement'''
    new_words = words.copy()
    random_word_list = list(set([word for word in words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        random_word = random_word.replace(' ', '')
        synonyms = get_synonyms(random_word, cwn)
        # print(random_word+'|', synonyms)
        
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            if verbose:
                print("replaced", random_word, "with", synonym)
            num_replaced += 1
        if num_replaced >= n: # only replace up to n words
            break
	# this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')
    return new_words

####### Random Deletion ######
def RandomDel(string, p):
    '''p: percentage'''
    if len(string) == 1: 
        return string
    new_string = []
    for s in string:
        r = random.uniform(0, 1)
        if r > p:
            new_string.append(s)
    if len(new_string) == 0:
        rand_int = random.randint(0, len(string)-1)
        return [string[rand_int]]
    return new_string

###### Random Swap ########
def RandomSwap(string, n):
    '''
    n: number of swaps performed
    '''
    new_words = string.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words
def swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words)-1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
      random_idx_2 = random.randint(0, len(new_words)-1)
      counter += 1
      if counter > 3:
        return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
    return new_words

############ Random Insertion ############ 
def RandomInsertion(words, n, cwn):
    if len(words) == 0:
      return words
    new_words = words.copy()
    for _ in range(n):
      add_word(new_words, cwn)
    return new_words

def add_word(new_words, cwn):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
      random_word = new_words[random.randint(0, len(new_words)-1)]
      random_word = random_word.replace(' ', '')
      synonyms = get_synonyms(random_word, cwn)
      # print(random_word+'|', synonyms)
      counter += 1
      if counter >= 10:
        return
    random_synonym = random.choice(synonyms)
    random_idx = random.randint(0, len(new_words)-1)
    new_words.insert(random_idx, random_synonym)      

############ EDA ###################
def eda(sentence, cwn, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug = 2):
    '''
    概念類似針對給定的一句sentence要對裡面多少percentage的字詞做這四種變換，如果α= 0.1，
    則該篇文章中10%的字詞會做同義詞變換、有10%的字詞會被刪除、10%的字會隨機換位，10%字詞的同義詞會
    被插入在隨機位置。
    '''
    '''
    input: a tokenized list of sentence
    return: [{num_aug} 數量的 augmented sentences (type: string), the original sentence (type: string)]
    '''
    cwn = cwn 
    words = RemoveDummies(sentence)
    num_words = len(sentence)
    num_new_per_technique = int(num_aug/4)+1
    augmented_sentences = []
  	#sr
    if (alpha_sr > 0): 
        n_sr = max(1, int(alpha_sr*num_words))
        for _ in range(num_new_per_technique):
            a_words = SynReplacement(words, n_sr, cwn)
            augmented_sentences.append(''.join(a_words))

	#ri
    if (alpha_ri > 0):
        n_ri = max(1, int(alpha_ri*num_words))
        for _ in range(num_new_per_technique):
            a_words = RandomInsertion(words, n_ri, cwn)
            augmented_sentences.append(''.join(a_words))

	#rs
    if (alpha_rs > 0):
        n_rs = max(1, int(alpha_rs*num_words))
        for _ in range(num_new_per_technique):
            a_words = RandomSwap(words, n_rs)
            augmented_sentences.append(''.join(a_words))

	#rd
    if (p_rd > 0):
        for _ in range(num_new_per_technique):
            a_words = RandomDel(words, p_rd)
            augmented_sentences.append(''.join(a_words))
    shuffle(augmented_sentences)

    #trim so that we have the desired number of augmented sentences
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

    # append the original sentence
    augmented_sentences.append(''.join(sentence)) 
    return augmented_sentences 

if __name__ == '__main__':
    main()



    

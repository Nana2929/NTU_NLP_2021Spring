'''
Cached words with lemmas that have been looked up before
Use a dictionary lookup if the word is in cache_dict
else find the word and add it into the cache_dict
'''
import json
def init():
    global cache_dict
    cache_dict = {}
def save_synonym_dict():
    with open('./synonym_dict', 'w', encoding = 'utf-8') as f:
        f.write(
            json.dumps(cache_dict,ensure_ascii=False, indent = 2))

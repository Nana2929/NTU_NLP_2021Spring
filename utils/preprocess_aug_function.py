def preprocess_augument_function(examples):
        question, option, texts = {}, {}, {}
        for i,label in enumerate(['A', 'B', 'C']): question[label] = [q['stem'] for q in examples['question']]
        for i,label in enumerate(['A', 'B', 'C']): option[label] = [q['choices'][i]['text'] for q in examples['question']]
        query_question = [jieba.lcut_for_search(q['stem']) for q in examples['question']]
        query_option = [jieba.lcut_for_search(q['choices'][0]['text']+'/'+q['choices'][1]['text']+'/'+q['choices'][2]['text']) for q in examples['question']]
        # print(query_question)
        # print(query_option)
        # print(examples['text'][0])
        for i in range(len(examples['text'])):
            examples['text'][i] = examples['text'][i].split('###')
            for j,text in enumerate(examples['text'][i]):
                examples['text'][i][j] = jieba.lcut_for_search(text)
            bm25 = BM25Okapi(examples['text'][i])
            doc_scores = bm25.get_scores(query_option[i])
            passage_count = len(examples['text'][i]) 
            reduce_count = int(passage_count * 0.1)
            _, retrieve_idx = map(list, zip(*sorted(zip(doc_scores, range(passage_count)),reverse=True)))
            # retrieve_idx = sorted(retrieve_idx[:reduce_count])
            retrieve_idx = sorted(retrieve_idx[:min(passage_count-1,5)])
            
            retrieve_passage = []
            for r in retrieve_idx: retrieve_passage.extend(examples['text'][i][r])
            examples['text'][i] = ''.join(retrieve_passage)
            while len(examples['text'][i]) < args.max_length*2: examples['text'][i] = examples['text'][i] + '/' + examples['text'][i]
        
        for label in ['A', 'B', 'C']: texts[label] = [p for p in examples['text']]
        def convert(c):
            if c == 'A': return 0
            elif c == 'B': return 1
            elif c == 'C': return 2
            else: 
                print(f'Invalid label "{c}"')
                exit()

        answers = list(map(convert, examples['answer']))
        tokenized_examples, tokenized_option = {}, {}
        for label in ['A', 'B', 'C']:
            tokenized_examples[label] = tokenizer(
                texts[label],
                max_length=args.max_length,
                truncation=True,
                stride=args.doc_stride,
                return_overflowing_tokens=True,
                padding = False,
            )
            tokenized_option[label] = tokenizer(
                option[label],
                stride=args.doc_stride,
                return_overflowing_tokens=True,
                padding = False,
            )
        sample_mapping = tokenized_examples['A']["overflow_to_sample_mapping"]
        inverted_file = {}
        for i,example_id in enumerate(sample_mapping):
            if example_id in inverted_file: inverted_file[example_id].append(i)
            else: inverted_file[example_id] = [i]
        for i in range(len(inverted_file)):
            inverted_file[i] = [inverted_file[i][0], inverted_file[i][1]]
            # passage_count = len(inverted_file[i])
            # print(inverted_file[i])
            # exit()
            # if passage_count > 4:
            #     tokenized_corpus = [tokenized_examples['A']['input_ids'][j] for j in inverted_file[i]]
            #     tokenized_corpus = tokenizer.batch_decode(tokenized_corpus, skip_special_tokens=True)
            #     bm25 = BM25Okapi(tokenized_corpus)
            #     # tokenized_query = [tokenized_option[j]['input_ids'][i] for j in ['A', 'B', 'C']]
            #     # tokenized_query = [t for option in tokenized_query for t in option]

            #     doc_scores = bm25.get_scores(query_option[i])
            #     reduce_count = int(passage_count * 0.9)
            #     _, inverted_file[i] = map(list, zip(*sorted(zip(doc_scores, inverted_file[i]),reverse=True)))
            #     inverted_file[i] = inverted_file[i][:reduce_count]
        selected_passages = sorted([i for _,lst in inverted_file.items() for i in lst])
        
        for label in ['A', 'B', 'C']:
            sample_mapping = tokenized_examples[label].pop("overflow_to_sample_mapping")
            option_len = len(tokenized_option[label]['input_ids'][0])
            tokenized_option[label]['input_ids'][0][0] = 102 # 102 [SEP]
            sample_mapping.append(-1)
            for i,sample_id in enumerate(sample_mapping):
                if sample_id == sample_mapping[i+1]:
                    tokenized_examples[label]['input_ids'][i] = tokenized_examples[label]['input_ids'][i][:-option_len]
                    tokenized_examples[label]['input_ids'][i].extend(tokenized_option[label]['input_ids'][sample_id])
                    tokenized_examples[label]['token_type_ids'][i] = tokenized_examples[label]['token_type_ids'][i][:-option_len+1]
                    for _ in range(option_len-1): tokenized_examples[label]['token_type_ids'][i].append(1)
                else:
                    paragraph_len = len(tokenized_examples[label]['input_ids'][i])
                    overflow_len = paragraph_len + option_len - 1 - args.max_length
                    if overflow_len > 0:    
                        tokenized_examples[label]['input_ids'][i] = tokenized_examples[label]['input_ids'][i][:-overflow_len-1]
                        tokenized_examples[label]['input_ids'][i].extend(tokenized_option[label]['input_ids'][sample_id])
                        tokenized_examples[label]['token_type_ids'][i] = tokenized_examples[label]['token_type_ids'][i][:-overflow_len]
                        for _ in range(option_len-1): tokenized_examples[label]['token_type_ids'][i].append(1)
                        tokenized_examples[label]['attention_mask'][i] = tokenized_examples[label]['attention_mask'][i][:-overflow_len-1]
                        tokenized_examples[label]['attention_mask'][i].extend(tokenized_option[label]['attention_mask'][sample_id])
                    else:
                        tokenized_examples[label]['input_ids'][i].pop(-1)
                        tokenized_examples[label]['input_ids'][i].extend(tokenized_option[label]['input_ids'][sample_id])
                        for _ in range(option_len-1): tokenized_examples[label]['token_type_ids'][i].append(1)
                        tokenized_examples[label]['attention_mask'][i].pop(-1)
                        tokenized_examples[label]['attention_mask'][i].extend(tokenized_option[label]['attention_mask'][sample_id])
                    if sample_mapping[i+1] == -1:
                        break
                    else:
                        option_len = len(tokenized_option[label]['input_ids'][sample_id+1])
                        tokenized_option[label]['input_ids'][sample_id+1][0] = 102
            sample_mapping.pop(-1)
        keys = tokenized_examples['A'].keys()
        tokenized_inputs = {k:[[tokenized_examples['A'][k][i],
                                tokenized_examples['B'][k][i],
                                tokenized_examples['C'][k][i]] for i in selected_passages] for k in keys}
        tokenized_inputs["labels"] = [] # 0 or 1 
        tokenized_inputs["example_id"] = []
        for i in selected_passages:
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_inputs["example_id"].append(examples["id"][sample_index])
            tokenized_inputs['labels'].append(answers[sample_index])
        

        for _ in range(2):
            for i in range(len(inverted_file)):
                a,b,c = np.random.choice(inverted_file[i],3)
                for k in keys:
                    tokenized_inputs[k].append([tokenized_examples['A'][k][a],
                                                tokenized_examples['B'][k][b],
                                                tokenized_examples['C'][k][c]])
                tokenized_inputs['labels'].append(answers[i])
                tokenized_inputs["example_id"].append(examples["id"][i])
        
        return tokenized_inputs
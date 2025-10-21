# pip3 install flask-restful
from flask import Flask, jsonify, request
import time
import numpy as np
from __main__ import app
import torch
from peft import AutoPeftModelForCausalLM
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSequenceClassification
from peft import PeftModel
import pandas as pd
import torch.nn.functional as F
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm.cli import tqdm
import numpy as np
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import random
from .vectordb import load_model, embed_search, embed_search_batch, tiktoken_len
from dotenv import load_dotenv
import os, re
import requests

load_dotenv()

# load model
print('Model Loading')

# Load BLOOM model
model_name = os.environ['LLM_BASE_MODEL'] # "E:\\Models\\bloom-3b"  # "bigscience/bloom-7b1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)



tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=False)
tokenizer.padding_side = 'right'
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     #load_in_8bit=True,
#     device_map="auto",
#     quantization_config=bnb_config,
# )


# if len(os.environ['LLM_STATE_DICT']) > 0:
#     print('>>>> Loading from state dict')
    
#     # Load lora model
#     def find_all_linear_names(model):
#         cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
#         lora_module_names = set()
#         for name, module in model.named_modules():
#             if isinstance(module, cls):
#                 names = name.split('.')
#                 lora_module_names.add(names[0] if len(names) == 1 else names[-1])

#         if 'lm_head' in lora_module_names:  # needed for 16-bit
#             lora_module_names.remove('lm_head')
#         return list(lora_module_names)
        
    
#     modules = find_all_linear_names(model)
#     peft_config = LoraConfig(
#         lora_alpha=32,
#         r=24,
#         target_modules=modules,
#         lora_dropout=0.1,
#         bias="none",
#         task_type="CAUSAL_LM",
#     )

#     model.gradient_checkpointing_enable()
#     model = prepare_model_for_kbit_training(model)
#     #modules = find_all_linear_names(model)
#     model = get_peft_model(model, peft_config)
#     model.load_state_dict(torch.load(os.environ['LLM_STATE_DICT'])) #"./nlp_api/models/bloom3b-checkpoint-160/model_state_dict_160.bin"))
# else:
#     print('>>>> Base model loaded.')

# #model = model.to('cuda')
# model.eval()
# model.config.use_cache = True


model_emb, tokenizer_emb, context_docs, context_embeddings = load_model(
    data_path=os.environ['DATA_DOC_PATHS'],    #'./nlp_api/data/Fixed/*',
    chunk_size=288, chunk_overlap=0, 
    separators=["\n\n", "\n", "।", "|", "?", ";", "!", ",", "-", "*", " ", ""]
)


with open(os.environ['PROMPT_SEARCHQ'], 'r') as f:
    searchq_prompt = f.read()
with open(os.environ['PROMPT_QA'], 'r') as f:
    ans_prompt = f.read()
print('\nModel loaded\n\n\n')



def get_context_from_prompt(query:str, max_gpt_tokens:int, max_bloom_tokens:int, shuffle_chunks:bool):
    result = embed_search(
        query, 
        docs=context_docs,
        embeddings=context_embeddings,
        model_emb=model_emb,
        tokenizer_emb=tokenizer_emb,
        topn=40
    )
    
    filtered_result = []
    gpt_token_len = 0
    bloom_token_len = 0
    for i in result:
        l = tiktoken_len(i['docs'].page_content + i['docs'].metadata['question'])
        if l + gpt_token_len > max_gpt_tokens:
            break
        
        bl = len(tokenizer.encode(i['docs'].page_content + i['docs'].metadata['question']))
        if bl + bloom_token_len > max_bloom_tokens:
            break
        
        filtered_result.append(i)
        gpt_token_len += l
        bloom_token_len += bl
    
    if shuffle_chunks:
        random.shuffle(filtered_result)
    return filtered_result, gpt_token_len, bloom_token_len




def get_contexts_from_prompt_batch(query:list, max_gpt_tokens:int, max_bloom_tokens:int, min_threshold=0, max_threshold=100, topn=15):  # nsamples=0, , ):
    result = embed_search_batch(
        query, 
        docs=context_docs,
        embeddings=context_embeddings,
        model_emb=model_emb,
        tokenizer_emb=tokenizer_emb,
        topn=topn
    )
    result = [x for x in result if x['score'] > min_threshold and x['score'] < max_threshold]
    
    final_results = []
    
    filtered_result = []
    gpt_token_len = 0
    bloom_token_len = 0
    for i in result:
        gl = tiktoken_len(i['docs'].page_content + i['docs'].metadata['question'])
        if gl + gpt_token_len > max_gpt_tokens:
            final_results.append(filtered_result)
            
            # Create new sample
            filtered_result = []
            gpt_token_len = 0
            bloom_token_len = 0
        
        bl = len(tokenizer.encode(i['docs'].page_content + i['docs'].metadata['question']))
        if bl + bloom_token_len > max_bloom_tokens:
            final_results.append(filtered_result)
            
            # Create new sample
            filtered_result = []
            gpt_token_len = 0
            bloom_token_len = 0
        
        filtered_result.append(i)
        gpt_token_len += gl
        bloom_token_len += bl
    
    return final_results



def generate(text, temperature, max_new_tokens):
    
    headers = {'Content-Type': 'application/json'}
    data = {
        "model": os.environ['LLM_BASE_MODEL_ENDPOINT'], #"./nlp_api/model_llama3/checkpoint-360-marged",
        "prompt": text,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
    }
    resp = requests.post('http://localhost:8000/v1/completions', json=data, headers=headers)
    if resp.status_code == 200:
        return resp.json()['choices'][0]['text']
    
    return None

# def generate(text, temperature, max_new_tokens):
#     # pipeline = transformers.pipeline(
#     #     "text-generation",
#     #     model=model,
#     #     tokenizer=tokenizer,
#     #     torch_dtype=torch.float16,
#     #     device_map="auto",
#     # )

#     # sequences = pipeline(
#     #     text,
#     #     do_sample=False,
#     #     #top_k=10,
#     #     num_return_sequences=1,
#     #     eos_token_id=tokenizer.eos_token_id,
#     #     max_new_tokens=780,
#     # )
#     # for seq in sequences:
#     #     print(seq['generated_text'])
    
#     # return sequences[0]['generated_text'].replace(text, '').strip()
    
    
#     encoded_input = tokenizer(text, return_tensors='pt')
#     with torch.no_grad():
#         output_sequences = model.generate(
#             input_ids=encoded_input['input_ids'].cuda(),
#             use_cache=True,
#             do_sample=True,
#             top_p=1.0,
#             temperature=temperature,
#             #eos_token_id=[105311, 21309],
#             #top_k=10,
#             #repetition_penalty=1.1,
#             #length_penalty=5+i*20,
#             max_new_tokens=max_new_tokens,
#         )
#         prompt_length = encoded_input['input_ids'].shape[1]
#         decoded = tokenizer.decode(output_sequences[0][prompt_length:], skip_special_tokens=True)
        
#         #decoded = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
#     return decoded.strip()
#     #return decoded.replace(text, '').strip()


def GenSearchQuery(query, query_ctx, temparature, max_new_tokens):
    
    input_text = searchq_prompt.replace('<history>', query_ctx).replace('<query>', query)
    print('>>>> SearchQ Input: ', input_text)
    
    response = generate(
        input_text, temperature=temparature, max_new_tokens=max_new_tokens,
    )
    print('>>>> SearchQ Response: ', response)
    
    
    statequery = re.findall(r'Step {step}:[\w\s&]+\n([\s\S\d]*?)\nStep'.format(step=1), response)[0]
    if 'Not Required' in response:
        print('>>>> Search Query not required')
        return statequery, [[]]
    
    response_tmp = response + '\n\nStep'
    q_en = re.findall(r'Step {step}:[\w\s&]+\n([\s\S\d]*?)\nStep'.format(step=3), response_tmp)[0].replace('-','').replace('"','').strip()
    q_en = q_en.split('\n')
    q_en = [x.strip() for x in q_en]
    #q_bn = re.findall(r'Step {step}:[\w\s&]+\n([\s\S\d]*?)\nStep'.format(step=3), response)[0].replace('-','').replace('"','').strip()
    
    print('>>>> Search: ', statequery, q_en)
    return statequery, [q_en]
    


def GenAnswer(query, query_ctx, statequery, context, temparature, max_new_tokens):
    input_text = ans_prompt.replace('<history>', query_ctx).replace('<query>', query).replace('<context>', context).replace('<statequery>', statequery)
    print('>>>> QA Input: ', input_text)
    
    response = generate(
        input_text, temperature=temparature, max_new_tokens=max_new_tokens,
    )
    print('>>>> QA Response: ', response)
    
    response_tmp = response + '\n\nStep'
    score = re.findall(r'Step {step}:[\w\s&]+\n([\s\S\d]*?)\nStep'.format(step=5), response_tmp)[0]
    answer = re.findall(r'Step {step}:[\w\s&]+\n([\s\S\d]*?)\nStep'.format(step=7), response_tmp)[0]
    answer = answer.replace('-','').strip()
    
    score = score.replace('-', '').strip()
    score = float(score)
    
    print('>>>> Score: ', score)
    print('>>>> Answer: ', answer)
    
    return score, answer



not_found_response = "দুঃখিত আমি আপনাকে এ ব্যাপারে সাহায্য করতে পারছি না।"

max_bloom_tokens = 1728
max_gpt_tokens = 2900

def chat_completion(dialog, model_name, mode_name, msg_context_size, ctx_checker_tmp, lm_tmp, topn, max_ctx, cls_threshold, llm_enable):
    log = {
        'retrived_docs': [], 'retrived_prob':[], 'matched_docs': [], 'matched_prob':[], 
        'llm_input': '', 'llm_reasoning':'', 'llm_response': '', 'module': ''
    }
    
    # Create message and message context
    query = dialog[-1]['content'].strip()
    msg_ctx = '\n'.join(x['content'] for x in dialog[-msg_context_size:-1])
    msg_ctx = 'None' if len(msg_ctx.strip()) == 0 else msg_ctx
    log['message_ctx'] = msg_ctx
    print('>>> msg_ctx: ', msg_ctx)
    
    
    # TODO Try multiple time if fails
    statequery, search_str = GenSearchQuery(
        query=query,
        query_ctx=msg_ctx,
        temparature=0.4,
        max_new_tokens=600,
    )
    
    # Only take english strings
    search_str = search_str[0]
    if len(search_str) == 0:
        print('>>>> No search string was generated')
        return not_found_response, log
    
    
    # Returns group of docs
    docs = get_contexts_from_prompt_batch(
        query=search_str, 
        max_gpt_tokens=max_gpt_tokens, 
        max_bloom_tokens=max_bloom_tokens,
        min_threshold=78,
        max_threshold=100,
        topn=10,
    )
    docs = docs[:2]   # Only try 4 times
    
    responses = []
    for d in docs:
        context = '\n---\n'.join([f"FAQ: {x['docs'].metadata['question']}\nAnswer: {x['docs'].page_content}" for x in d])
        
        score, answer = GenAnswer(
            query=query,
            query_ctx=msg_ctx,
            statequery=statequery,
            context=context,
            temparature=0,
            max_new_tokens=1200,
        )
        
        responses.append({'score': score, 'answer': answer, 'docs': d})
        
        if score >= 3.5:
            break
        continue
        
        if score < 4.0:
            print('>>>> Low Score: ', score)
            continue
        
        # Log
        # TODO Input of Search Query
        # TODO Input of QA
        log['retrived_prob'] = [x['score'] for x in d]
        log['retrived_docs'] = [f"FAQ: {x['docs'].metadata['question']}\nAnswer: {x['docs'].page_content}" for x in d]
        log['matched_docs'] = log['retrived_docs']
        log['matched_prob'] = log['retrived_prob']
        #log['llm_reasoning'] = response
        log['llm_response'] = answer
        return answer, log
    
    
    responses = sorted(responses, key=lambda x: x['score'], reverse=True)
    print('>>>> Responses:', responses)
    
    if len(responses) == 0 or responses[0]['score'] < 1.6:
        return not_found_response, log
    
    
    # Get best answer
    fanswer_dict = responses[0]
    fanswer = fanswer_dict['answer']
    ans_docs = fanswer_dict['docs']
    # Log
    # TODO Input of Search Query
    # TODO Input of QA
    log['retrived_prob'] = [x['score'] for x in ans_docs]
    log['retrived_docs'] = [f"FAQ: {x['docs'].metadata['question']}\nAnswer: {x['docs'].page_content}" for x in ans_docs]
    log['matched_docs'] = log['retrived_docs']
    log['matched_prob'] = log['retrived_prob']
    #log['llm_reasoning'] = response
    log['llm_response'] = fanswer
    return fanswer, log



@app.route('/llm/v1/api', methods = ['POST'])
def llm_infer():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        
        ## Need to validate json
        # need to chat history. must not be empty
        
        cls_threshold = 0.33
        ctx_checker_tmp = 0.008
        lm_tmp = 0.12
        max_ctx = 3
        llm_enable = True
        if 'config' in json:
            ctx_checker_tmp = json['config']['ctx_checker_tmp']
            lm_tmp = json['config']['lm_tmp']
            max_ctx = json['config']['max_ctx']
            
            cls_threshold = json['config']['cls_threshold']
            llm_enable = json['config']['llm_enable']
            
            print('Debug Mode:  ', ctx_checker_tmp, lm_tmp)
        
        
        ## Run Model
        print('Transcription Started')
        resp, log = chat_completion(
            json['chat_history'], 
            model_name=json['model'], 
            mode_name=json['mode'],
            msg_context_size=5,   # -1 is the actual message context size
            ctx_checker_tmp=ctx_checker_tmp,
            lm_tmp=lm_tmp,
            topn = 24,
            max_ctx=max_ctx,
            cls_threshold=cls_threshold,
            llm_enable = llm_enable,
        )
        print(resp)
        
        data = {
            'responses': [
                {'role': 'ai', 'content': resp, 'meta': log['module']}
            ],
            'refer_needed': False,
            'logs': {
                'version': '1',
                'content': {
                    'context_llm': {'generated_ctx': ''},
                    'retrival_model': {
                        'retrived_doc': log['retrived_docs'][:16],
                        'retrived_prob': [float(x) for x in log['retrived_prob']][:16],
                        'matched_doc': log['matched_docs'],
                        'matched_prob': [float(x) for x in log['matched_prob']]
                    },
                    'llm': {
                        'response': log['llm_response'],
                        'reasoning': log['llm_reasoning'],
                        'input': log['llm_input'],
                        'message_ctx': log['message_ctx']
                    },
                    'risk_llm': {
                        
                    },
                    'risk_detection': {
                        
                    },
                    'sop_module': {
                        
                    },  
                }
            }
        }
        return jsonify({'data': data})
    return 'Content-Type not supported!', 400


  
# driver function
if __name__ == '__main__':
    app.run(debug = True)

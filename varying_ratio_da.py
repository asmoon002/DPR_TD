import json
import re
import random
import psutil
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from transformers import BertTokenizer
from collections import Counter
import torch
from nltk.stem import PorterStemmer
import gc
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet', download_dir='./nltk_data')
nltk.data.path.append('./nltk_data')
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# Initialize device and stemmer
#device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
epsilon=1e-8
prefix_pattern = re.compile(r"^(un|re|in|im|dis|non|over|mis)")
suffix_pattern = re.compile(r'(ess|ment|ity|tion|sion|able|ible|al|ial|ic|ous|ive|en|ize|ise|ate|fy|ly|ing|ed|er|est)$')

def nltk_to_wordnet_pos(nltk_pos):
    if nltk_pos.startswith('J'):
        return wn.ADJ  # Adjective
    elif nltk_pos.startswith('V'):
        return wn.VERB  # Verb
    elif nltk_pos.startswith('N'):
        return wn.NOUN  # Noun
    else:
        return None  # Ignore other POS types

# Validate if a prefix is real based on WordNet
def is_real_prefix(prefix, stem):
    return bool(wn.synsets(stem))

def load_json(path):
    with open(path, "r") as file:
        return json.load(file)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as writer:
        json.dump(data, writer, ensure_ascii=False, indent=4)

def chunk_data(data, chunk_size=2000):  # Smaller chunk size: 2000
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

def tokenize_doc(doc, tokenizer):
    return tokenizer.tokenize(doc[0])

def tokenize_documents(docs, tokenizer, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(tqdm(executor.map(lambda doc: tokenize_doc(doc, tokenizer), docs), total=len(docs), mininterval=1.0, dynamic_ncols=True))

def is_valid_word(word):
    """Check if the word is a valid word in WordNet."""
    return bool(wn.synsets(word))

# POS-aware stem validator
def is_valid_stem(stem_candidate):
    noun_synsets = wn.synsets(stem_candidate.lower(), pos=wn.NOUN)
    verb_synsets = wn.synsets(stem_candidate.lower(), pos=wn.VERB)
    adj_synsets = wn.synsets(stem_candidate.lower(), pos=wn.ADJ)
    return bool(noun_synsets) or bool(verb_synsets) or bool(adj_synsets)


def process_tokens(doc):
    processed_tokens = []
    for token in doc:
        current_token = token

       
        prefix_match = prefix_pattern.match(current_token)
        if prefix_match:
            prefix = prefix_match.group()
            remaining_stem = current_token[len(prefix):]
            if is_valid_stem(remaining_stem) and len(remaining_stem) > 2:
                processed_tokens.append(prefix)
                current_token = remaining_stem

        
        suffix_match = suffix_pattern.search(current_token)
        if suffix_match and len(current_token) > 1:
            suffix = suffix_match.group()
            stem_candidate = current_token[:-len(suffix)]

            # Try lemmatizer first
            lemma_candidate = lemmatizer.lemmatize(current_token, pos='n')
            if lemma_candidate == current_token:
                lemma_candidate = lemmatizer.lemmatize(current_token, pos='v')

            # Use lemmatizer result if valid
            if lemma_candidate != current_token and is_valid_stem(lemma_candidate) and len(lemma_candidate) > 2:
                processed_tokens.append(lemma_candidate)
                processed_tokens.append("##" + suffix)
                continue

            # Fallback to original stem_candidate if valid
            if is_valid_stem(stem_candidate) and len(stem_candidate) > 2:
                processed_tokens.append(stem_candidate)
                processed_tokens.append("##" + suffix)
                continue
            else:
                processed_tokens.append(current_token)
                continue


        # Fallback to lemmatizer for inflectional suffixes
        lemma = lemmatizer.lemmatize(current_token, pos='v')
        if lemma == current_token:
            lemma = lemmatizer.lemmatize(current_token, pos='n')
        if lemma == current_token:
            lemma = lemmatizer.lemmatize(current_token, pos='a')

        if lemma != current_token and is_valid_stem(lemma):
            suffix_len = len(current_token) - len(lemma)
            suffix = current_token[-suffix_len:] if suffix_len > 0 else ''
            processed_tokens.append(lemma)
            if suffix:
                processed_tokens.append("##" + suffix)
        else:
            # No change → keep token as is
            processed_tokens.append(current_token)

    return processed_tokens


def softmax(x):
    y = 1/x
    return np.exp(y) / np.sum(np.exp(y))

def calculate_weights(doc_tokens, df_counts, alpha=1.0, beta=1.0, scale_factor=0.2):
    weights = []
    for doc_token in doc_tokens:
        PF = doc_tokens.count(doc_token)
        CF = df_counts[doc_token]
        combined_weight = (PF ** alpha) * (CF ** beta)

        weights.append({'token': doc_token, 'weight': combined_weight})

    weight_values = np.array([w['weight'] for w in weights])
    max_weight = weight_values.max()
    min_weight = weight_values.min()
    diff = (max_weight - min_weight) + epsilon
    scaled_weights = [{'token': weights[i]['token'], 'weight': (weight_values[i] - min_weight) / diff} for i in range(len(weights))]

    final_weights = [{'token': scaled_weights[i]['token'], 'weight': scaled_weights[i]['weight'] * scale_factor} for i in range(len(scaled_weights))]

    return final_weights




def reconstruct_sentence(data, base_mask_prob=0.1):
    tokens = [item['token'] for item in data['weights']]
    sentence = []
    for item in data['weights']:
        token = item['token']
        weight = item['weight']
        # Adjust mask probability based on weight
        
        #mask        
        current_token = "[mask]" if random.random() < weight else token
        
        #deletion
        #current_token = "" if random.random() < weight else token
        
        if current_token.startswith("##"):
            if sentence:
                sentence[-1] = sentence[-1]+current_token[2:]
        else:
            sentence.append(current_token)
    return ' '.join(sentence)

def process_chunk(data_chunk, tokenizer):
    documents = []
    positive_ctx_indices = []

    # Collect only positive contexts
    for item in data_chunk:
        for ctx_index, ctx in enumerate(item['positive_ctxs']):
            documents.append([ctx['text']])
            positive_ctx_indices.append((item, ctx_index))

    tokenized_documents_temp = tokenize_documents(documents, tokenizer)
    tokenized_documents = [process_tokens(doc) for doc in tokenized_documents_temp]





    df_counts = Counter([token for doc_tokens in tokenized_documents for token in set(doc_tokens)])

    tokenized_results = []
    for doc_index, doc_tokens in enumerate(tokenized_documents):
        ############이번엔 여기가 수정!!!!!!!!!!!!!!!!!!!!!!!!!!
        weights = calculate_weights(doc_tokens, df_counts, alpha=XXX, beta=XXX, scale_factor=XXX)
        tokenized_results.append({
            "document_index": doc_index,
            "weights": weights,
        })



    save_result = [reconstruct_sentence(d) for d in tokenized_results]




    # Update only the positive_ctxs in the original data_chunk
    for i, reconstructed_sentence in enumerate(save_result):
        item, ctx_index = positive_ctx_indices[i]
        item['positive_ctxs'][ctx_index]['text'] = reconstructed_sentence

    # Clear memory
    del documents, tokenized_documents_temp, tokenized_documents, df_counts, tokenized_results
    gc.collect()

    return save_result, data_chunk

def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(f"Memory usage: {mem_info.rss / (1024 ** 2):.2f} MB")

def main():
    try:
        json_path = "fil_path/WikiQA_train_processed.json"
        data_loaded = load_json(json_path)
        chunks = chunk_data(data_loaded)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        for j, chunk in enumerate(chunks):
            
            logger.info(f"Processing chunk {j+1}/{len(chunks)}")
            save_result, modified_data = process_chunk(chunk, tokenizer)
            
            #여기 폴더 이름 두개 바꾸셔야합니다. no_using은 사용하지 않아요
            #pro_aug 1~N 개 까지 json으로 만들고 aggregator로 학습 데이터셋을 취합합니다.
            #save_json(save_result, f'save_path/no_using{j+1}.json')
            save_json(modified_data, f'save_path/pro_aug{j+1}.json')
            logger.info(f"Chunk {j+1} processed and saved successfully")
            log_memory_usage()  # Log memory usage after processing each chunk
            
                
            

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

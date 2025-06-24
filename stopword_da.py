import json
import re
import psutil
import logging
from tqdm import tqdm
import gc
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', download_dir='./nltk_data')
nltk.data.path.append('./nltk_data')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load stopword set
stopwords_set = set(stopwords.words('english'))

# Load / Save utils
def load_json(path):
    with open(path, "r") as file:
        return json.load(file)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as writer:
        json.dump(data, writer, ensure_ascii=False, indent=4)

# Chunk utils
def chunk_data(data, chunk_size=2000):
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

# Stopword-only text processing (단순 whitespace split)
def process_text_stopword(text, stopwords_set):
    tokens = text.split()
    filtered_tokens = [token for token in tokens if token.lower() not in stopwords_set]
    return ' '.join(filtered_tokens)

# Process chunk → stopword-only 증강
def process_chunk_stopword_simple(data_chunk, stopwords_set):
    positive_ctx_indices = []

    # Collect positive_ctxs
    for item in data_chunk:
        for ctx_index, ctx in enumerate(item['positive_ctxs']):
            positive_ctx_indices.append((item, ctx_index))

    # Apply stopword deletion
    for item, ctx_index in tqdm(positive_ctx_indices, desc="Stopword Deletion", dynamic_ncols=True):
        orig_text = item['positive_ctxs'][ctx_index]['text']
        new_text = process_text_stopword(orig_text, stopwords_set)
        item['positive_ctxs'][ctx_index]['text'] = new_text

    gc.collect()

    # Just return updated data_chunk
    return data_chunk

# Memory log
def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(f"Memory usage: {mem_info.rss / (1024 ** 2):.2f} MB")

# Main
def main():
    try:
        json_path = "fil_path/WikiQA_train_processed.json"
        data_loaded = load_json(json_path)
        chunks = chunk_data(data_loaded)
        
        for j, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {j+1} [Stopword baseline]")
            modified_data = process_chunk_stopword_simple(chunk, stopwords_set)

            save_json(modified_data, f'save_path/stopword_da{j+1}.json')
            logger.info(f"Chunk {j+1} processed and saved successfully [Stopword baseline]")
            log_memory_usage()

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

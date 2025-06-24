import json
import os
import gc
from tqdm import tqdm
# 트레이닝 데이터를 저장할 빈 json 파일을 만듭니다
empty_file_path = 'file_path/empty_file_name.json'
with open(empty_file_path, 'w') as empty_file:
    json.dump([], empty_file)

def append_json_files(target_file, json_files):
    # Read the current content of the target file
    with open(target_file, 'r') as file:
        data = json.load(file)
    
    # Append data from each json file in the list
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        if os.path.exists(json_file):
            with open(json_file, 'r') as file:
                new_data = json.load(file)
                data.extend(new_data)
            # Delete the data from memory to free up memory

            del new_data
            gc.collect()
        else:
            print(f"File {json_file} does not exist.")
    
    # Write the updated data back to the target file
    with open(target_file, 'w') as file:
        json.dump(data, file, indent=4)

# Example usage:
# Assuming we have a list of JSON files to append
json_files_to_append = [f'파일 경로/da로 만들어진 파일 이름{i}.json' for i in range(1, 55)]
# Add the additional JSON file path to the list
additional_file = '파일 경로/final_train.json'
json_files_to_append.append(additional_file) # Replace with actual file paths
append_json_files(empty_file_path, json_files_to_append)



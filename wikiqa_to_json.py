import json
import pandas as pd

def preprocess_wikiqa_updated(data):
    grouped_data = {}
    
    for _, row in data.iterrows():
        question_id = row['QuestionID']
        if question_id not in grouped_data:
            grouped_data[question_id] = {
                "dataset": "wikiqa",
                "question": row['Question'],
                "answers": [],
                "positive_ctxs": [],
                "negative_ctxs": []
            }
        
        # Append relevant data based on the label
        context = {
            "title": row['DocumentTitle'],
            "text": row['Sentence'],
        }
        if row['Label'] == 1:
            grouped_data[question_id]["answers"].append(row['Sentence'])
            grouped_data[question_id]["positive_ctxs"].append(context)
        else:
            grouped_data[question_id]["negative_ctxs"].append(context)
    
    # Filter questions without positive contexts or with multiple positives
    filtered_data = []
    for question_id, content in grouped_data.items():
        if content["positive_ctxs"] and content["negative_ctxs"]:  # Ensure both positive and negative contexts exist
            # Keep only the first positive context and its associated answer
            content["positive_ctxs"] = content["positive_ctxs"][:1]
            content["answers"] = content["answers"][:1]
            filtered_data.append(content)
    
    return filtered_data

# Load and process the data
input_file_path = '/home/kyumin/jbig/dataset/WikiQA-dev.tsv'  # Replace with your file path
data = pd.read_csv(input_file_path, sep='\t')

# Process the data with the updated function
processed_data_updated = preprocess_wikiqa_updated(data)

# Save the updated processed data to a JSON file
output_file_path = '/home/kyumin/jbig/dataset/WikiQA_dev_processed.json'  # Replace with your desired output path
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(processed_data_updated, f, ensure_ascii=False, indent=4)

print(f"Processed data has been saved to: {output_file_path}")

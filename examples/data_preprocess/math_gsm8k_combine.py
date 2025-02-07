import requests
import json
import re
import os
import datasets
import argparse


def extract_solution(answer_str: str) -> str:
    """
    Try to extract a solution using a regex pattern.
    If the answer string contains a substring like "#### <solution>",
    extract the numeric solution; otherwise, return the stripped answer.
    """
    m = re.search(r"####\s*(\-?[0-9\.\,]+)", answer_str)
    if m:
        # Remove commas from the number, if any.
        return m.group(1).replace(',', '')
    else:
        return answer_str.strip()

def format_as_deepseek_prompt(question_raw: str) -> str:
    """
    Create a prompt similar to the DeepSeek prompt.
    """
    full_prompt = (
        "A conversation between User and Assistant. The user asks a question, "
        "and the Assistant solves it. The assistant first thinks about the reasoning process "
        "in the mind and then provides the user with the answer. The reasoning process and answer "
        "are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think>\n"
        "<answer> answer here </answer>. User: " + question_raw + ". Assistant:"
    )
    return full_prompt

def extract_data_from_json(url: str, filter_by_level=[1, 2, 3]):
    """
    Downloads the JSON file from the given URL and filters items by the specified levels.
    """
    response = requests.get(url)
    processed_data = []

    if response.status_code == 200:
        data = response.json()  # Parse JSON content
        for item in data:
            if item["level"] in filter_by_level:
                processed_data.append({
                    "question": item["question"],
                    "gt_answer": item["gt_answer"],
                    "level": item["level"]
                })
    else:
        print(f"Error fetching data. Status code: {response.status_code}")
    return processed_data

def make_map_fn(split: str, answer_field: str, data_source: str):
    """
    Returns a function to process each example.
    
    Args:
        split: A string indicating the data split ('train' or 'test').
        answer_field: The field name in the raw example that holds the answer.
        data_source: Identifier for the data source.
    """
    def process_fn(example, idx):
        # Extract and format the question.
        question_raw = example.pop('question')
        question = format_as_deepseek_prompt(question_raw)
        
        # Extract the answer using the provided answer field.
        answer_raw = example.pop(answer_field)
        solution = extract_solution(answer_raw)
        
        return {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": question,
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer_raw,
                "question": question_raw,
            }
        }
    return process_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/gsm8k')
    parser.add_argument('--hdfs_dir', default=None)
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir

    # Define the common data_source identifier.
    data_source = 'deepseek/gsm8k'

    ##############################################
    # Process Dataset 1: JSON-based dataset
    ##############################################
    json_url = "https://raw.githubusercontent.com/hkust-nlp/simpleRL-reason/refs/heads/main/train/data/math_level3to5_data_processed_with_qwen_prompt.json"
    processed_data = extract_data_from_json(json_url)
    
    # Create a Hugging Face dataset from the JSON data.
    dataset_json = datasets.Dataset.from_list(processed_data)
    # Split into train/test splits (80/20 split).
    split_dataset_json = dataset_json.train_test_split(test_size=0.2, seed=42)
    train_json = split_dataset_json['train']
    test_json = split_dataset_json['test']
    
    # Process the JSON dataset.
    # For this dataset, the answer is stored in the "gt_answer" field.
    train_json = train_json.map(function=make_map_fn('train', 'gt_answer', data_source), with_indices=True)
    test_json = test_json.map(function=make_map_fn('test', 'gt_answer', data_source), with_indices=True)

    # Print one example from the JSON-based datasets.
    print("JSON-based train example:")
    print(train_json[0])
    print("\nJSON-based test example:")
    print(test_json[0])

    ##############################################
    # Process Dataset 2: GSM8k from Hugging Face
    ##############################################
    # Load the GSM8k dataset.
    dataset_hf = datasets.load_dataset('openai/gsm8k', 'main')
    train_hf = dataset_hf['train']
    test_hf = dataset_hf['test']
    
    # Process the GSM8k dataset.
    # For this dataset, the answer is stored in the "answer" field.
    train_hf = train_hf.map(function=make_map_fn('train', 'answer', data_source), with_indices=True)
    test_hf = test_hf.map(function=make_map_fn('test', 'answer', data_source), with_indices=True)

    # Print one example from the GSM8k datasets.
    print("\nGSM8k train example:")
    print(train_hf[0])
    print("\nGSM8k test example:")
    print(test_hf[0])

    ##############################################
    # Combine the two datasets
    ##############################################
    combined_train = datasets.concatenate_datasets([train_json, train_hf])
    combined_test = datasets.concatenate_datasets([test_json, test_hf])

    # Save the combined datasets as parquet files.
    combined_train_path = os.path.join(local_dir, 'combined_train.parquet')
    combined_test_path = os.path.join(local_dir, 'combined_test.parquet')
    combined_train.to_parquet(combined_train_path)
    combined_test.to_parquet(combined_test_path)
    
    print(f"\nSaved combined train dataset to {combined_train_path}")
    print(f"Saved combined test dataset to {combined_test_path}")

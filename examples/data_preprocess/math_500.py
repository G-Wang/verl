import requests
import json
import re
import os
import datasets
import argparse

# URL of the JSON file
url = "https://raw.githubusercontent.com/hkust-nlp/simpleRL-reason/refs/heads/main/train/data/math_level3to5_data_processed_with_qwen_prompt.json"


def extract_data_from_json(url, filter_by_level=[1, 2, 3]):
    """
    Downloads the JSON file from the given URL and filters items
    by the specified levels.
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


def extract_solution(answer_raw: str) -> str:
    """
    Dummy implementation of extract_solution.
    You can adjust this function to extract the desired solution from the raw answer.
    """
    return answer_raw.strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/gsm8k')
    parser.add_argument('--hdfs_dir', default=None)
    args = parser.parse_args()

    processed_data = extract_data_from_json(url)

    # Create a Hugging Face dataset from the processed JSON data
    dataset = datasets.Dataset.from_list(processed_data)
    # Split the dataset into train and test splits (80/20 split)
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']

    # Use a new data_source identifier since we're not using the original deepseek/gsm8k dataset.
    data_source = 'deepseek/gsm8k'

    # Define a mapping function to process each example
    def make_map_fn(split):
        def process_fn(example, idx):
            # Extract and format the question
            question_raw = example.pop('question')
            question = format_as_deepseek_prompt(question_raw)

            # Use the 'gt_answer' field from the JSON data
            answer_raw = example.pop('gt_answer')
            solution = extract_solution(answer_raw)

            data = {
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
                    'split': split,
                    'index': idx,
                    'answer': answer_raw,
                    "question": question_raw,
                }
            }
            return data

        return process_fn

    # Process each example in the train and test splits
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # Expand the user's home directory if needed
    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir

    # Save the datasets as parquet files
    train_path = os.path.join(local_dir, 'train.parquet')
    test_path = os.path.join(local_dir, 'test.parquet')
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)
    print(f"Saved train dataset to {train_path}")
    print(f"Saved test dataset to {test_path}")

    # Print out a single example from the training dataset for verification
    print("Sample train example:")
    print(train_dataset[0])

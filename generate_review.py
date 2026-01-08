import json
import os
import random
import argparse
import re
from datasets import load_dataset
from ai_scientist.perform_review import perform_review
import time
from datetime import datetime
import openai
import dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
# Set CUDA device if needed
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# Load environment variables
dotenv.load_dotenv()

# Configuration
RANDOM_SEED = 42

# Set random seed for reproducibility
random.seed(RANDOM_SEED)


def extract_boxed_review(text):
    """Extract content from \\boxed_review{ }"""
    if not text:
        return text
    pattern = r'\\boxed_review\{(.*?)\}'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate reviews using OpenAI API on WestlakeNLP/DeepReview-13K dataset"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt-4.1",
        help="OpenAI model name to use (default: gpt-4.1)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to evaluate. Use -1 to evaluate all data (default: 100)"
    )
    parser.add_argument(
        "--num-reflections",
        type=int,
        default=1,
        help="Number of reflections for review generation (default: 1)"
    )
    parser.add_argument(
        "--num-fs-examples",
        type=int,
        default=1,
        help="Number of few-shot examples (default: 1)"
    )
    parser.add_argument(
        "--num-reviews-ensemble",
        type=int,
        default=1,
        help="Number of reviews in ensemble (default: 1)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of parallel workers for API calls (default: 5)"
    )
    return parser.parse_args()


def main(args):

    # 1. Load WestlakeNLP/DeepReview-13K dataset (test split only)
    print("Loading DeepReview-13K dataset (test split)...")
    ds = load_dataset("WestlakeNLP/DeepReview-13K", split="test")

    # 2. Extract paper context from data and set as an extra column named as paper_text
    print("Extracting paper text from dataset...")

    def extract_paper_text(example):
        """Extract paper content from the user role in inputs"""
        # The inputs field contains a list of messages
        # Extract the content from the message with role "user"
        inputs = example['inputs']
        paper_text = ""

        # If inputs is a string, parse it as JSON
        if isinstance(inputs, str):
            try:
                inputs = json.loads(inputs)
            except json.JSONDecodeError:
                inputs = []

        if isinstance(inputs, list):
            for message in inputs:
                if isinstance(message, dict) and message['role'] == 'user':
                    paper_text = message['content']
                    break

        example['paper_text'] = paper_text
        return example

    # Apply extraction to test split
    ds = ds.map(extract_paper_text)

    # 3. Append all paper_text to a list
    print("Collecting all paper texts...")
    paper_texts = list(ds['paper_text'])

    # Randomly sample papers with seed for reproducibility
    if args.num_samples == -1:
        print(f"Using all {len(paper_texts)} papers")
    elif args.num_samples > 0 and len(paper_texts) > args.num_samples:
        # Get random indices
        sampled_indices = random.sample(range(len(paper_texts)), args.num_samples)
        sampled_indices.sort()  # Sort to maintain some order
        paper_texts = [paper_texts[i] for i in sampled_indices]
        # Also need to filter the dataset to match
        ds = ds.select(sampled_indices)
        print(f"Sampled {len(paper_texts)} papers (seed: {RANDOM_SEED})")
    else:
        print(f"Using all {len(paper_texts)} papers (num_samples={args.num_samples} >= dataset size)")

    # 4. Initialize OpenAI client and run evaluations
    print(f"\nInitializing OpenAI client with model: {args.model_name}...")
    client = openai.OpenAI()

    # Start timing
    start_time = time.time()
    print(f"\nGenerating reviews for {len(paper_texts)} papers...")
    print(f"Parameters: num_reflections={args.num_reflections}, num_fs_examples={args.num_fs_examples}, num_reviews_ensemble={args.num_reviews_ensemble}")
    print(f"Using {args.max_workers} parallel workers")

    # Thread-safe counter and lock for progress tracking
    completed_count = 0
    lock = Lock()

    def process_paper(idx, paper_text):
        """Process a single paper and return its review"""
        nonlocal completed_count
        try:
            review = perform_review(
                paper_text,
                args.model_name,
                client,
                num_reflections=args.num_reflections,
                num_fs_examples=args.num_fs_examples,
                num_reviews_ensemble=args.num_reviews_ensemble,
            )
            with lock:
                completed_count += 1
                print(f"Completed paper {completed_count}/{len(paper_texts)} (idx: {idx + 1})")
            return idx, review
        except Exception as e:
            with lock:
                completed_count += 1
                print(f"Error processing paper {idx + 1}: {e}")
            return idx, {"error": str(e)}

    # Process papers in parallel using ThreadPoolExecutor
    review_results = [None] * len(paper_texts)  # Pre-allocate list to maintain order

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_paper, idx, paper_text): idx
            for idx, paper_text in enumerate(paper_texts)
        }

        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx, review = future.result()
            review_results[idx] = review

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nReview generation completed in {elapsed_time:.2f} seconds")
    print(f"Average time per paper: {elapsed_time/len(paper_texts):.2f} seconds")

    print("\nBenchmark completed!")
    print(f"Successfully generated {len([r for r in review_results if r and 'error' not in r])} reviews")

    # 5. Prepare output data with all required fields
    print("\nPreparing output data...")
    print("Verifying index correspondence...")

    # Verify that indices match
    mismatches = 0
    for i in range(len(paper_texts)):
        if ds[i]['paper_text'] != paper_texts[i]:
            print(f"WARNING: Index mismatch at position {i}")
            mismatches += 1

    if mismatches == 0:
        print(f"✓ Index verification passed: all {len(paper_texts)} entries match correctly")
    else:
        print(f"✗ Found {mismatches} index mismatches!")

    output_data = []
    for i in range(len(paper_texts)):
        entry = {
            'id': ds[i]['id'],
            'title': "",
            'paper_context': ds[i]['paper_text'],
            'decision': ds[i]['decision'],
            'human_review': ds[i]['reviewer_comments'],
            'golden_review': extract_boxed_review(json.loads(ds[i]['outputs'])[2]['content']),
            'model_prediction': {"raw_text": review_results[i]},
        }
        output_data.append(entry)

    # 6. Save to JSON file
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # Clean model name for filename (e.g., "gpt-4.1" -> "gpt-4-1")
    model_filename = args.model_name.replace('/', '-').replace('.', '-')
    output_dir = 'evaluate/review'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/openai_{model_filename}_sample_{args.num_samples}_{timestamp}.json'
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Results saved successfully! Total entries: {len(output_data)}")

    return ds, paper_texts, review_results, output_data


if __name__ == "__main__":
    args = parse_args()
    dataset, papers, review_results, output_data = main(args)

"""
Example usage:

# Generate reviews for all papers using GPT-4.1 (default, 5 parallel workers)
python generate_review.py --model-name gpt-4.1 --num-samples 100 --max-workers 32
python generate_review.py --model-name gpt-4o --num-samples 100 --max-workers 32
"""
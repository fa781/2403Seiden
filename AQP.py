
'''
import json
import random
import numpy as np

def load_index(path="output/full_index.json"):
    with open(path, "r") as f:
        return json.load(f)["all_frames"]

def select_top_k(data, k):
    return sorted(data, key=lambda x: x['score'], reverse=True)[:k]

def sample_by_importance(data, k):
    total_score = sum(d['score'] for d in data)
    if total_score == 0:
        return random.sample(data, k)
    probabilities = [d['score'] / total_score for d in data]
    return random.choices(data, weights=probabilities, k=k)

def filter_by_score(data, threshold=0.5):
    return [d for d in data if d['score'] > threshold]

def stratified_sample(data, bins=5, samples_per_bin=2):
    data = sorted(data, key=lambda x: x['score'])
    scores = [d['score'] for d in data]
    bin_edges = np.linspace(min(scores), max(scores), bins + 1)
    result = []
    for i in range(bins):
        bin_items = [d for d in data if bin_edges[i] <= d['score'] < bin_edges[i + 1]]
        if bin_items:
            result.extend(random.sample(bin_items, min(len(bin_items), samples_per_bin)))
    return result

def run_queries(index_path, k, threshold):
    data = load_index(index_path)
    print("Top-K Frames:")
    print(select_top_k(data, k))

    print("\nRandom Sampling by Importance:")
    print(sample_by_importance(data, k))

    print("\nFiltered Frames (score > threshold):")
    print(filter_by_score(data, threshold))

    print("\nStratified Sample by Score:")
    print(stratified_sample(data, bins=5, samples_per_bin=4))

if __name__ == "__main__":
    run_queries()
'''


import json
import random
import numpy as np
import os
from PIL import Image

def load_index(path="output/full_index.json"):
    with open(path, "r") as f:
        return json.load(f)["all_frames"]

def select_top_k(data, k):
    return sorted(data, key=lambda x: x['score'], reverse=True)[:k]

def sample_by_importance(data, k):
    total_score = sum(d['score'] for d in data)
    if total_score == 0:
        return random.sample(data, k)
    probabilities = [d['score'] / total_score for d in data]
    return random.choices(data, weights=probabilities, k=k)

def filter_by_score(data, threshold=0.5):
    return [d for d in data if d['score'] > threshold]

def stratified_sample(data, bins=5, samples_per_bin=2):
    data = sorted(data, key=lambda x: x['score'])
    scores = [d['score'] for d in data]
    bin_edges = np.linspace(min(scores), max(scores), bins + 1)
    result = []
    for i in range(bins):
        bin_items = [d for d in data if bin_edges[i] <= d['score'] < bin_edges[i + 1]]
        if bin_items:
            result.extend(random.sample(bin_items, min(len(bin_items), samples_per_bin)))
    return result

def save_results(results_dict, output_path="output/aqp_results.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=4)

def save_top_frame_image(top_frame_index, frame_dir="output/frames", output_path="output/top_frame.jpg"):
    input_path = os.path.join(frame_dir, f"frame_{top_frame_index}.jpg")
    if os.path.exists(input_path):
        image = Image.open(input_path)
        image.save(output_path)
    else:
        print(f"Warning: Frame image {input_path} does not exist.")

def run_queries(index_path="output/full_index.json", frame_dir="output/frames", k=20, threshold=0.5):
    data = load_index(index_path)

    top_k = select_top_k(data, k)
    importance_sampled = sample_by_importance(data, k)
    thresholded = filter_by_score(data, threshold)
    stratified = stratified_sample(data, bins=5, samples_per_bin=4)

    results = {
        "top_k": top_k,
        "importance_sampled": importance_sampled,
        "thresholded": thresholded,
        "stratified": stratified
    }

    save_results(results)

    # Print and save top result frame index
    if top_k:
        top_frame_index = top_k[0]['frame_index']
        print(f"Top result frame index: {top_frame_index}")
        save_top_frame_image(top_frame_index, frame_dir)
    else:
        print("No top-k frames available.")

if __name__ == "__main__":
    run_queries()

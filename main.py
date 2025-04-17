import os
from input_processing import input_processing
from index_construction import construct_index
from MAB_sampling import MAB_Sampling
from label_propagation import propagate_labels
from AQP import run_queries

def main():
    
    # Define paths and parameters
    video_path = "./videoInput/hong_kong_airport_demo_data.mp4"        # Input video
    output_frame_dir = "./output/frames"            # Output Directory
    model_budget = 40                                   # Total budget on how many frames going through the model 
    sampling_ratio = 0.8                        # Anchor Ratio: Fraction of frames to sample
    query = "a suitcase."                          # Query for GroundingDINO testing
    output = "./output/constructed_index.json"      # Path for output index JSON file
    full_index_path="output/full_index.json"

    # Ensure output directory exists
    os.makedirs(output_frame_dir, exist_ok=True)

    # Step 1 input processing
    print("Running input_processing...")
    sampled_frames, sampled_frame_indices, sampled_timestamps, full_iframe_indices, full_frame_indices = input_processing(
        video_path, output_frame_dir, sampling_ratio
    )

    # Print results of Step 1
    print("\n--- Input Processing Results ---")
    print(f"Number of sampled frames: {len(sampled_frames)}")
    print(f"Sampled frame indices: {sampled_frame_indices}")

    # Step 2 index construction
    constructed_index = construct_index(
        sampled_frames, sampled_frame_indices, query, output
    )
    print(f"Index construction completed. Constructed index has {len(constructed_index['sampled_frames'])} entries.")

    # Step 3 MAB Sampling
    already_processed = len(sampled_frames)  # frames already processed during index construction
    remaining_budget = model_budget - already_processed
    if remaining_budget <= 0:
        print("Model budget exhausted after index construction.")
    else:
        print(f"Remaining model budget for MAB Sampling: {remaining_budget}")
        MAB_Sampling(full_iframe_indices, sampled_frame_indices, constructed_index, query, n_samples=remaining_budget, c_param=2)

    # Step 4 label propagation
    full_index = propagate_labels(constructed_index, full_frame_indices)

    # Run SUPG AQP query
    print("Running SUPG for AQP...")
    selected_indices = run_queries(full_index_path, k=20, threshold=0.5)


    
if __name__ == "__main__":
    main()

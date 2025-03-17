import os
from input_processing import input_processing
from index_construction import construct_index
from MAB_sampling import MAB_Sampling

def main():
    
    # Define paths and parameters
    video_path = "./videoInput/video360.mp4"        # Input video
    output_frame_dir = "./output/frames"            # Output Directory
    model_budget = 50                                     # Total budget on how many frames going through the model 
    sampling_ratio = 0.1                         # Anchor Ratio: Fraction of frames to sample
    query = "a human."                          # Query for GroundingDINO testing
    output = "./output/constructed_index.json"      # Path for output index JSON file

    # Ensure output directory exists
    os.makedirs(output_frame_dir, exist_ok=True)

    # Step 1 input processing
    print("Running input_processing...")
    sampled_frames, sampled_frame_indices, sampled_timestamps, full_iframe_indices = input_processing(
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

    # Step 3: Perform MAB Sampling (Active Learning)
    already_processed = len(sampled_frames)  # frames already processed during index construction
    remaining_budget = model_budget - already_processed
    if remaining_budget <= 0:
        print("Model budget exhausted after index construction.")
    else:
        print(f"Remaining model budget for MAB Sampling: {remaining_budget}")
        # Step 4: MAB Sampling using the remaining budget
        MAB_Sampling(full_iframe_indices, sampled_frame_indices, constructed_index, query, n_samples=remaining_budget, c_param=2)

    
if __name__ == "__main__":
    main()

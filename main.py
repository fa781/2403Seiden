import os
from input_processing_savingFrames import input_processing  # Import the input_processing function
#from preprocessing import preprocess_frames
from index_construction import construct_index


def main():
    """
    Main function to test the input processing functionality.
    """
    # Define paths and parameters
    video_path = "./videoInput/video360.mp4"            # Input video
    output_frame_dir = "./output/frames"    # Output Directory
    sampling_ratio = 0.01          # Fraction of frames to sample
    fps = 30  # Input video frames per second
    query = "Find all objects"
    output = "./output/constructed_index.json"

    # Ensure output directory exists
    os.makedirs(output_frame_dir, exist_ok=True)

    # Step 1 input processing
    # Call the input_processing function
    print("Running input_processing...")
    sampled_frames, sampled_frame_indices, sampled_timestamps = input_processing(
        video_path, output_frame_dir, sampling_ratio
    )

    # Print results
    print("\n--- Input Processing Results ---")
    print(f"Number of sampled frames: {len(sampled_frames)}")
    print(f"Sampled frame indices: {sampled_frame_indices}")
    # print(f"Sampled frame timestamps: {sampled_timestamps}")
    print(f"Sampled frame timestamps (in seconds): {[timestamp / fps for timestamp in sampled_timestamps]}")

    '''
    preprocessed_frames = preprocess_frames(sampled_frames)

    print("\n--- Preprocessing Results ---")
    print(f"Number of preprocessed frames: {len(preprocessed_frames)}")    
    '''
    # Step 2 index construction
    print("Running index construction...")
    constructed_index = construct_index(sampled_frames, sampled_frame_indices, sampled_timestamps, query, output)

    print("Index construction completed.")
    print(f"Constructed index has {len(constructed_index)} entries.")


if __name__ == "__main__":
    main()
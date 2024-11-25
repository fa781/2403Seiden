
import torch
from transformers import GroundingDinoForObjectDetection, AutoProcessor
import gc
import psutil
import json
import os
'''
def construct_index(frames, frame_indices, timestamps, query, outputJSON):
    """
    Constructs an index from sampled video frames using the GroundingDINO model.

    Parameters:
        frames (list of numpy.ndarray): Sampled video frames in RGB format.
        frame_indices (list of int): Indices of sampled frames.
        timestamps (list of float): Corresponding timestamps for sampled frames.
        query (str): Query text for the GroundingDINO model.

    Returns:
        dict: The constructed index containing detected objects and metadata.
    """
    print("Running index construction...")

    try:
        print("Loading GroundingDINO model and processor...")
        model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
        processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        print("Model and processor loaded successfully.")
    except Exception as e:
        print("Error loading GroundingDINO model or processor. Check the model identifier and authentication.")
        raise e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Default query if not provided
    if not query:
        query = "Find all objects"
    print(f"Query: {query}")

    # Initialize the index
    constructed_index = {
        "frames": [],
        "objects": [],
        "features": [],
    }

    # Batch size (adjust based on GPU memory)
    batch_size = 1  
    print(f"Batch size set to: {batch_size}")

    try:
        for batch_start in range(0, len(frames), batch_size):
            batch_end = min(batch_start + batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]
            batch_indices = frame_indices[batch_start:batch_end]
            batch_timestamps = timestamps[batch_start:batch_end]

            print(f"Processing batch starting at frame {batch_start}...")

            # Preprocess frames
            inputs = processor(images=batch_frames, text=query, return_tensors="pt")
            inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to device

            # Run inference
            outputs = model(**inputs)

            # Process outputs
            for i, index in enumerate(batch_indices):
                print(f"Processing frame {index} (timestamp: {batch_timestamps[i]:.2f}s)")
                try:
                    logits = outputs.logits[i].detach().cpu().numpy()
                    boxes = outputs.pred_boxes[i].detach().cpu().numpy()

                    # Append to index
                    constructed_index["frames"].append(index)
                    constructed_index["objects"].append({
                        "logits": logits.tolist(),
                        "boxes": boxes.tolist(),
                    })
                    constructed_index["features"].append({
                        "timestamp": batch_timestamps[i],
                        "frame_index": index,
                    })
                    print(f"Appending index: {index}, logits length: {len(logits)}, boxes length: {len(boxes)}")

                except Exception as e:
                    print(f"Error processing frame {index} (timestamp: {batch_timestamps[i]:.2f}s). Skipping... Error: {str(e)}")

            # Clear memory
            del inputs, outputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except RuntimeError as e:
        print(f"An error occurred during index construction: {type(e).__name__} - {str(e)}")

    print("Index construction completed.")
    print(f"Constructed index contains {len(constructed_index['frames'])} entries.")

    with open("constructed_index_details.txt", "w") as file:
        file.write(str(constructed_index))
    print("Index details saved to constructed_index_details.txt")

    # Save the index to a JSON file
    output_dir = os.path.dirname(outputJSON)
    if output_dir:  # Only create directories if there's a valid path
        os.makedirs(output_dir, exist_ok=True)
        with open(outputJSON, "w") as f:
            json.dump(index, f, indent=4)
        print(f"Index saved to {outputJSON}")
    return constructed_index
'''



def construct_index(frames, frame_indices, timestamps, query, outputJSON):
    """
    Constructs an index from sampled video frames using the GroundingDINO model.

    Parameters:
        frames (list of numpy.ndarray): Sampled video frames in RGB format.
        frame_indices (list of int): Indices of sampled frames.
        timestamps (list of float): Corresponding timestamps for sampled frames.
        query (str): Query text for the GroundingDINO model.

    Returns:
        dict: The constructed index containing detected objects and metadata.
    """
    print("Running index construction...")

    try:
        print("Loading GroundingDINO model and processor...")
        model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
        processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        print("Model and processor loaded successfully.")
    except Exception as e:
        print("Error loading GroundingDINO model or processor. Check the model identifier and authentication.")
        raise e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Default query if not provided
    if not query:
        query = "Find all objects"
    print(f"Query: {query}")

    # Initialize the index
    constructed_index = {
        "frames": [],
        "objects": [],
        "features": [],
    }

    # Batch size (adjust based on GPU memory)
    batch_size = 1  
    print(f"Batch size set to: {batch_size}")

    try:
        for batch_start in range(0, len(frames), batch_size):
            batch_end = min(batch_start + batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]
            batch_indices = frame_indices[batch_start:batch_end]
            batch_timestamps = timestamps[batch_start:batch_end]

            print(f"Processing batch starting at frame {batch_start}...")

            # Preprocess frames
            inputs = processor(images=batch_frames, text=query, return_tensors="pt")
            inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to device

            # Run inference
            outputs = model(**inputs)

            # Process outputs
            for i, index in enumerate(batch_indices):
                print(f"Processing frame {index} (timestamp: {batch_timestamps[i]:.2f}s)")
                try:
                    logits = outputs.logits[i].detach().cpu().numpy()
                    boxes = outputs.pred_boxes[i].detach().cpu().numpy()

                    # Debugging output
                    # print(f"Debug - Frame {index}: Logits shape: {logits.shape}, Boxes shape: {boxes.shape}")
                    # print(f"Sample logits: {logits[:5]}")  # Print a small sample for inspection
                    # print(f"Sample boxes: {boxes[:5]}")  # Print a small sample for inspection

                    # Append to index
                    constructed_index["frames"].append(index)
                    constructed_index["objects"].append({
                        "logits": logits.tolist(),
                        "boxes": boxes.tolist(),
                    })
                    constructed_index["features"].append({
                        "timestamp": batch_timestamps[i],
                        "frame_index": index,
                    })
                    print(f"Appending index: {index}, logits length: {len(logits)}, boxes length: {len(boxes)}")

                except Exception as e:
                    print(f"Error processing frame {index} (timestamp: {batch_timestamps[i]:.2f}s). Skipping... Error: {str(e)}")

            # Clear memory
            del inputs, outputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except RuntimeError as e:
        print(f"An error occurred during index construction: {type(e).__name__} - {str(e)}")

    print("Index construction completed.")
    print(f"Constructed index contains {len(constructed_index['frames'])} entries.")

    # with open("constructed_index_details.txt", "w") as file:
    #     file.write(str(constructed_index))
    # print("Index details saved to constructed_index_details.txt")

    # Save the index to a JSON file
    output_dir = os.path.dirname(outputJSON)
    if output_dir:  # Only create directories if there's a valid path
        os.makedirs(output_dir, exist_ok=True)
        with open(outputJSON, "w") as f:
            json.dump(constructed_index, f, indent=4)
        print(f"Index saved to {outputJSON}")
    return constructed_index
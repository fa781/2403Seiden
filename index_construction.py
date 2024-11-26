import torch
from transformers import GroundingDinoForObjectDetection, AutoProcessor
import gc
import json
import os

def construct_index(frames, frame_indices, timestamps, query, outputJSON):
    """
    Constructs an index from sampled video frames using the GroundingDINO model.

    Parameters:
        frames (list of numpy.ndarray): Sampled video frames in RGB format.
        frame_indices (list of int): Indices of sampled frames.
        timestamps (list of float): Corresponding timestamps for sampled frames.
        query (str): Query text for the GroundingDINO model.
        outputJSON (str): Path to save the constructed index in JSON format.

    Returns:
        dict: The constructed index containing detected objects and metadata.
    """
    print("Running index construction...")

    # Load model and processor
    print("Loading GroundingDINO model and processor...")
    model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    print("Model and processor loaded successfully.")

    constructed_index = {}

    for i, frame in enumerate(frames):
        frame_index = frame_indices[i]
        timestamp = timestamps[i]

        print(f"Processing frame {frame_index} ...")
        try:
            # Preprocess frame
            inputs = processor(images=[frame], text=query, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Run inference
            outputs = model(**inputs)
            
            # Extract logits and boxes
            logits = outputs.logits[0].detach().cpu().numpy()
            boxes = outputs.pred_boxes[0].detach().cpu().numpy()
            
            # Map logits to probabilities and COCO labels
            probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
            detected_objects = {}
            for j, box in enumerate(boxes):
                label_idx = logits[j].argmax()
                label = processor.tokenizer.decode([label_idx])  # Adjust to match COCO label extraction
                probability = probabilities[j][label_idx]
                
                if probability > 0.35:  # Filter low-confidence detections
                    detected_objects[label] = float(probability)

            # Add to index
            constructed_index[frame_index] = {
                "frame index": frame_index,
                "objects": detected_objects,
            }

            print(f"Frame {frame_index}: {len(detected_objects)} objects detected.")
        
        except Exception as e:
            print(f"Error processing frame {frame_index}: {e}")

        # Clear GPU memory
        del inputs, outputs
        torch.cuda.empty_cache()

    # Save constructed index to JSON
    os.makedirs(os.path.dirname(outputJSON), exist_ok=True)
    with open(outputJSON, "w") as f:
        json.dump(constructed_index, f, indent=4)
    print(f"Index saved to {outputJSON}")

    return constructed_index

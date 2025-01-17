import json
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def construct_index(sampled_frames, frame_indices, timestamps, query, outputJSON):
    # Model setup
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # Dictionary to store results
    constructed_index = {"sampled_frames": []}

    # Process each frame
    for i, frame in enumerate(sampled_frames):
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(frame)

        # Prepare inputs for Grounding DINO
        inputs = processor(images=pil_image, text=query, return_tensors="pt").to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Post-process results
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[pil_image.size[::-1]]
        )

        # Extract boxes and scores for this frame
        boxes = results[0]["boxes"].tolist() if "boxes" in results[0] else []
        frame_scores = results[0]["scores"].tolist() if "scores" in results[0] else []

        # Append results for this frame to the dictionary
        constructed_index["sampled_frames"].append({
            "frame_index": frame_indices[i],
            "timestamp": timestamps[i],
            "scores": frame_scores,
            "boxes": boxes
        })

    # Save results to JSON file
    with open(outputJSON, "w") as json_file:
        json.dump(constructed_index, json_file, indent=4)

    print(f"Results saved to {outputJSON}")

    return constructed_index
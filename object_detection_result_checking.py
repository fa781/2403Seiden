import os
import json
from PIL import Image, ImageDraw

def plot_bounding_boxes(input_dir, json_path, output_dir, label):
    """
    Annotate images with bounding boxes and save the results.

    :param input_dir: Directory containing input frames.
    :param json_path: Path to the JSON file with detection results.
    :param output_dir: Directory to save annotated images.
    :param label: Label to annotate frames with (default: "Human")
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load JSON results
    with open(json_path, 'r') as file:
        detection_results = json.load(file)

    # Iterate through each frame's results
    for frame_result in detection_results["sampled_frames"]:
        frame_index = frame_result["frame_index"]
        scores = frame_result.get("scores", [])
        boxes = frame_result.get("boxes", [])  # Ensure this key exists in the JSON structure
        
        input_image_path = os.path.join(input_dir, f"frame_{frame_index}.jpg")
        output_image_path = os.path.join(output_dir, f"frame_{frame_index}_annotated.jpg")

        # Load the corresponding image
        if not os.path.exists(input_image_path):
            print(f"Frame {frame_index} not found in {input_dir}. Skipping...")
            continue

        image = Image.open(input_image_path)
        draw = ImageDraw.Draw(image)

        # Annotate image with bounding boxes and scores
        for i, box in enumerate(boxes):
            x_min, y_min, x_max, y_max = box
            score = scores[i]

            # Draw bounding box
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

            # Add label and score
            text = f"{label}: {score:.2f}"
            draw.text((x_min, y_min - 10), text, fill="red")

        # Save the annotated image
        image.save(output_image_path)
        print(f"Annotated image saved to {output_image_path}")

if __name__ == "__main__":
    # Define paths
    input_frames_dir = "./output/frames"  # Directory containing original frames
    json_results_path = "./output/constructed_index.json"  # Path to JSON results
    output_annotated_dir = "./output/annotated_frames"  # Directory for annotated frames

    # Run the annotation
    plot_bounding_boxes(input_frames_dir, json_results_path, output_annotated_dir, "Human")

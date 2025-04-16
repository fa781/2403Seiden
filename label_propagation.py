import json
import numpy as np

def propagate_labels(constructed_index, full_frame_indices):
    """
    Propagate proxy scores from sampled frames (anchors) to all frames in the video,
    using linear interpolation based on frame indices.
    
    :param constructed_index: Dictionary from the previous index construction/MAB sampling.
                              Expected to have a key "sampled_frames", where each entry is a dict with:
                                  "frame_index": int,
                                  "scores": list of floats.
    :param full_frame_indices: List of all frame indices in the video (e.g., list(range(total_frames))).
    :return: new_index, a dictionary with key "all_frames" that contains an entry for every frame.
             Each entry includes the frame index and the propagated score.
    """
    
    # Extract anchor frames (sampled frames) and sort by frame_index.
    anchors = sorted(constructed_index["sampled_frames"], key=lambda x: x["frame_index"])
    
    # Build a dictionary mapping anchor frame index to a representative score (average of scores).
    anchor_dict = {}
    for entry in anchors:
        idx = entry["frame_index"]
        scores = entry.get("scores", [])
        # Use the average score as the representative score. If no scores, default to 0.
        rep_score = float(np.mean(scores)) if scores else 0.0
        anchor_dict[idx] = rep_score
    
    # Get a sorted list of anchor indices.
    anchor_indices = sorted(anchor_dict.keys())
       
    full_index = {"all_frames": []}
    
    for frame in full_frame_indices:
        if frame in anchor_dict:
            # Use the anchor score directly.
            entry = {
                "frame_index": frame,
                "score": anchor_dict[frame],
                "propagated": False  # Indicates that this frame was processed by the model.
            }
        else:
            # Find the nearest anchor frames.
            prev_anchor = None
            next_anchor = None
            for a in anchor_indices:
                if a <= frame:
                    prev_anchor = a
                elif a > frame:
                    next_anchor = a
                    break
            
            if prev_anchor is None:
                # No previous anchor; use next anchor.
                interp_score = anchor_dict[next_anchor]
            elif next_anchor is None:
                # No next anchor; use previous anchor.
                interp_score = anchor_dict[prev_anchor]
            else:
                # Linear interpolation based on frame indices.
                score1 = anchor_dict[prev_anchor]
                score2 = anchor_dict[next_anchor]
                frac = (frame - prev_anchor) / float(next_anchor - prev_anchor)
                interp_score = score1 + frac * (score2 - score1)
            
            entry = {
                "frame_index": frame,
                "score": interp_score,
                "propagated": True  # Indicates that this score was interpolated.
            }
            
        full_index["all_frames"].append(entry)
    
    # Sort full_index by frame index.
    full_index["all_frames"].sort(key=lambda x: x["frame_index"])
    
    # Save the full index to a JSON file.
    with open("output/full_index.json", "w") as f:
        json.dump(full_index, f, indent=4)
    print("Full index with propagated scores saved to output/full_index.json")
    
    return full_index

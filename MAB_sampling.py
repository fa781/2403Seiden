import json
import numpy as np
import os
from PIL import Image
from MAB_utils import ClusterManager
from index_construction import construct_index

def MAB_Sampling(i_frame_indices, sampled_frame_indices, constructed_index, query, n_samples=50, c_param=2):
    """
    Perform MAB sampling to iteratively select frames using the UCB strategy and process them with Grounding DINO.
    """
    cluster_manager = ClusterManager()
    cluster_manager.initialize_clusters(i_frame_indices, sampled_frame_indices, constructed_index)
    
    for iteration in range(n_samples):
        total_samples = len(cluster_manager.sampled_frames) + 1

        # Compute UCB scores
        ucb_values = cluster_manager.compute_ucb(c_param, total_samples)


        # Debug: print all UCB scores for each cluster
        print(f"\nIteration {iteration+1}:")
        '''
        print("UCB values for clusters:")
        for cluster, score in ucb_values.items():
            print(f"  Cluster {cluster}: {score}")
        '''
                
        try:
            selected_cluster = cluster_manager.select_cluster(ucb_values)
            print(f"Selected cluster: {selected_cluster}")

            sampled_frame = cluster_manager.sample_from_cluster(selected_cluster)

            if sampled_frame is None:
                print(f"No valid unvisited frames found in cluster {selected_cluster}. Skipping iteration.")
                continue
            else:
                print(f"Sampled frame {sampled_frame} from cluster {selected_cluster}.")

            # Load the image corresponding to the sampled frame index
            image_path = f"./output/frames/frame_{sampled_frame}.jpg"
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} does not exist. Skipping this frame.")
                continue

            pil_image = Image.open(image_path)
            frame_array = np.array(pil_image)

            # Run Grounding DINO
            print(f"Calling construct_index for frame {sampled_frame}... Running Grounding DINO.")
            results = construct_index(
                sampled_frames=[frame_array], 
                frame_indices=[sampled_frame],  
                query=query,
                outputJSON="output/constructed_index.json"
            )

            # Ensure results are valid before updating the index
            if "sampled_frames" in results and results["sampled_frames"]:
                new_frame_data = results["sampled_frames"][-1]
                print(f"Appending new frame {new_frame_data['frame_index']} to constructed index.")
                constructed_index["sampled_frames"].append(new_frame_data)
            else:
                print(f"Warning: No detection results for frame {sampled_frame}. Skipping update.")

            # Update cluster metadata
            new_frame_scores = results["sampled_frames"][-1]["scores"] if results["sampled_frames"] else []
            cluster_manager.update_cluster(selected_cluster, sampled_frame, new_frame_scores)

        except ValueError:
            print("No valid clusters available. Stopping MAB sampling.")
            break

    # Save the updated constructed index
    print("Saving updated constructed index to file...")
    with open("output/constructed_index.json", "w") as f:
        json.dump(constructed_index, f, indent=4)
    print("Save completed.")

    print("MAB Sampling completed. Updated index saved.")

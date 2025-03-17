from collections import OrderedDict
import numpy as np
import random

class ClusterManager:
    def __init__(self):
        self.cluster_dict = OrderedDict()
        self.sampled_frames = []

    def initialize_clusters(self, i_frame_indices, sampled_frame_indices, constructed_index):
        """
        Initialize clusters by dividing I-frame indices using sampled frame indices as boundaries and
        populate cache and distance using scores from constructed_index.

        :param i_frame_indices: List of all I-frame indices from the video.
        :param sampled_frame_indices: List of sampled I-frame indices.
        :param constructed_index: Dictionary containing scores and metadata for sampled frames.
        """
        self.cluster_dict = OrderedDict()

        sampled_frame_indices = sorted(sampled_frame_indices)
        self.sampled_frames = list(sampled_frame_indices)   # modified, not firmed yet
        i_frame_indices = sorted(i_frame_indices)

        # Build a lookup for frame scores from constructed_index
        frame_scores_lookup = {
            entry["frame_index"]: entry["scores"]
            for entry in constructed_index["sampled_frames"]
        }

        for i in range(len(sampled_frame_indices)):
            start_boundary = sampled_frame_indices[i]
            end_boundary = sampled_frame_indices[i + 1] if i + 1 < len(sampled_frame_indices) else None

            # Collect frames in the current cluster
            members = [
                frame for frame in i_frame_indices
                if frame >= start_boundary and (end_boundary is None or frame < end_boundary)
            ]

            # Populate cache with scores if available, otherwise use placeholders
            cache = [
                frame_scores_lookup.get(frame, [0])  # Default to [0] if no scores are found
                for frame in members
            ]
            flattened_cache = [score for sublist in cache for score in sublist]  # Flatten the nested lists

            # Calculate distance (variance of cache)
            distance = np.var(flattened_cache) if flattened_cache else 0.0

            # Populate the cluster dictionary
            self.cluster_dict[(start_boundary, end_boundary or float('inf'))] = {
                "members": members,
                "cache": flattened_cache,
                "distance": distance
            }

    def compute_ucb(self, c_param, total_samples):
        """
        Compute UCB values for all clusters.
        """
        ucb_values = {}

        for cluster_key, cluster_data in self.cluster_dict.items():
            reward = cluster_data["distance"]
            cluster_size = len(cluster_data["members"])

            unvisited_frames = [
                frame for frame in cluster_data["members"]
                if frame not in self.sampled_frames
            ]

            if cluster_size == 0 or len(unvisited_frames) == 0:
                ucb_values[cluster_key] = float('-inf')  # Exclude depleted clusters
            else:
                ucb_values[cluster_key] = reward + c_param * np.sqrt(2 * np.log(total_samples) / cluster_size)

        return ucb_values

    def select_cluster(self, ucb_values):
        valid_clusters = {key: value for key, value in ucb_values.items() if value != float('-inf')}
        if not valid_clusters:
            raise ValueError("No valid clusters available for selection.")
        return max(valid_clusters, key=valid_clusters.get)

    def sample_from_cluster(self, selected_cluster):
        """
        Sample a frame from the selected cluster. If the cluster has only one frame,
        print a message and do nothing.
        """
        unvisited_frames = [
            frame for frame in self.cluster_dict[selected_cluster]["members"]
            if frame not in self.sampled_frames
        ]

        if not unvisited_frames:
            raise ValueError(f"No unvisited frames available in cluster {selected_cluster}.")

        sampled_frame = random.choice(unvisited_frames)
        return sampled_frame

    def update_cluster(self, selected_cluster, sampled_frame, frame_scores):
        """
        Update the metadata of the selected cluster after sampling a frame.
        """
        self.cluster_dict[selected_cluster]["members"].append(sampled_frame)
        self.cluster_dict[selected_cluster]["cache"].extend(frame_scores)
        flattened_scores = self.cluster_dict[selected_cluster]["cache"]
        self.cluster_dict[selected_cluster]["distance"] = np.var(flattened_scores) if flattened_scores else 0.0
        self.sampled_frames.append(sampled_frame)

    def mark_cluster_depleted(self, selected_cluster):
        """
        Mark the selected cluster as depleted by setting its distance to 0
        and clearing its members.
        """
        self.cluster_dict[selected_cluster]["distance"] = 0
        self.cluster_dict[selected_cluster]["members"] = []

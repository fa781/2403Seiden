import os
import numpy as np
from src.iframes.ffmpeg_commands import FfmpegCommands
from src.iframes.pyav_utils import PYAV_wrapper
from PIL import Image

def input_processing(video_path, output_dir, sampling_ratio):
    """
    Processes input video to extract sampled frames based on a given sampling ratio.

    :param video_path: Path to the input video file
    :param output_dir: Directory to save extracted frames
    :param sampling_ratio: Fraction of I-frames to sample (1.0 = all frames)
    :return: sampled_frames, sampled_frame_indices, sampled_timestamps
    """

    # Initialize helpers
    ffmpeg_helper = FfmpegCommands()
    pyav_helper = PYAV_wrapper(video_path)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract I-frame indices using both FFmpeg and PyAV
    print("Extracting I-frame indices...")
    iframe_indices_ffmpeg = ffmpeg_helper.get_iframe_indices(video_path)
    # iframe_indices_pyav = pyav_helper.get_iframe_indices()

    # check output from iframe detectors
    # print(iframe_indices_ffmpeg)
    # print(iframe_indices_pyav)

    # Reconcile I-frame indices
    # reconciled_indices = np.union1d(iframe_indices_ffmpeg, iframe_indices_pyav)
    reconciled_indices = iframe_indices_ffmpeg
    print(f"Reconciled I-frame indices: {reconciled_indices.tolist()}")

    # Load all keyframes
    print("Loading keyframes...")
    keyframes, keyframe_timestamps = pyav_helper.load_keyframes()

    # Determine sampled frames based on sampling ratio
    num_samples = int(len(reconciled_indices) * sampling_ratio)
    # sampled_indices = np.linspace(0, len(reconciled_indices) - 1, num_samples, dtype=int)
    sampled_indices = np.random.choice(len(reconciled_indices), num_samples, replace=False) #randomly picked
    sampled_frame_indices = reconciled_indices[sampled_indices]
    sampled_frames = keyframes[sampled_indices]
    sampled_timestamps = [keyframe_timestamps[i] for i in sampled_indices]

    print(f"Sampling ratio: {sampling_ratio}, Total Samples: {num_samples}")

    # Save all I-frames
    for frame, frame_id in zip(keyframes, reconciled_indices):
        img = Image.fromarray(frame)
        output_path = os.path.join(output_dir, f"frame_{frame_id}.jpg")
        if not os.path.exists(output_path):
            # print(f"Saving frame {frame_id} to {output_path}")
            img.save(output_path)

    # Get the full video length (total number of frames)
    total_frames = pyav_helper.get_video_length()
    full_frame_indices = list(range(total_frames))
    print(f"Total number of frames in video: {total_frames}")

    return sampled_frames, sampled_frame_indices.tolist(), sampled_timestamps, reconciled_indices.tolist(), full_frame_indices
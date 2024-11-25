# import os
# import numpy as np
# from src.iframes.pyav_utils import PYAV_wrapper
# from src.iframes.ffmpeg_commands import FfmpegCommands
# import cv2


import os
import numpy as np
from src.iframes.origin_ffmpeg_commands import FfmpegCommands
from src.iframes.origin_pyav_utils import PYAV_wrapper
from PIL import Image

def input_processing(video_path, output_dir, sampling_ratio):
    """
    Processes input video to extract sampled frames based on a given sampling ratio.

    :param video_path: Path to the input video file
    :param output_dir: Directory to save extracted frames
    :param sampling_ratio: Fraction of I-frames to sample (1.0 = all frames)
    :return: sampled_frames, sampled_frame_indices, sampled_timestamps
    """
    print("Running input_processing...")

    # Initialize helpers
    ffmpeg_helper = FfmpegCommands()
    pyav_helper = PYAV_wrapper(video_path)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract I-frame indices using both FFmpeg and PyAV
    print("Extracting I-frame indices...")
    iframe_indices_ffmpeg = ffmpeg_helper.get_iframe_indices(video_path)
    iframe_indices_pyav = pyav_helper.get_iframe_indices()

    # Reconcile I-frame indices
    reconciled_indices = np.intersect1d(iframe_indices_ffmpeg, iframe_indices_pyav)
    print(f"Reconciled I-frame indices: {reconciled_indices.tolist()}")

    # Load all keyframes
    print("Loading keyframes...")
    keyframes, keyframe_timestamps = pyav_helper.load_keyframes()

    # Determine sampled frames based on sampling ratio
    num_samples = int(len(reconciled_indices) * sampling_ratio)
    sampled_indices = np.linspace(0, len(reconciled_indices) - 1, num_samples, dtype=int)
    sampled_frame_indices = reconciled_indices[sampled_indices]
    sampled_frames = keyframes[sampled_indices]
    sampled_timestamps = [keyframe_timestamps[i] for i in sampled_indices]
    # sampled_timestamps = [index / fps for index in sampled_frame_indices]  # Convert to seconds

    print(f"Sampling ratio: {sampling_ratio}, Total Samples: {num_samples}")

    # Save sampled frames
    for i, frame in enumerate(sampled_frames):
        # Convert numpy array to PIL Image
        img = Image.fromarray(frame)
        output_path = os.path.join(output_dir, f"frame_{i}.jpg")
        print(f"Saving frame {i} to {output_path}")
        img.save(output_path)  # Save the image using PIL

    return sampled_frames, sampled_frame_indices.tolist(), sampled_timestamps

'''
def input_processing(video_path, output_frame_dir, sampling_ratio):
    os.makedirs(output_frame_dir, exist_ok=True)

    pyav_helper = PYAV_wrapper(video_path)
    ffmpeg_helper = FfmpegCommands()

    print("Extracting video length and I-frame indices...")
    total_frames = pyav_helper.get_video_length()
    print(f"Total frames: {total_frames}")

    iframe_indices_pyav = pyav_helper.get_iframe_indices()
    iframe_indices_ffmpeg = ffmpeg_helper.get_iframe_indices(video_path)

    print(f"PyAV I-frame indices: {iframe_indices_pyav}")
    print(f"FFmpeg I-frame indices: {iframe_indices_ffmpeg}")

    iframe_indices = sorted(set(iframe_indices_pyav).intersection(iframe_indices_ffmpeg))
    print(f"Reconciled I-frame indices: {iframe_indices}")

    keyframes, timestamps = pyav_helper.load_keyframes()
    print(f"Loaded keyframes: {len(keyframes)}")

    n_samples = max(1, int(len(iframe_indices) * sampling_ratio))
    print(f"Sampling ratio: {sampling_ratio}, Samples: {n_samples}")

    sampled_frame_indices = iframe_indices[:n_samples]
    valid_indices = [i for i in sampled_frame_indices if i < len(keyframes)]
    print(f"Valid sampled indices: {valid_indices}")

    sampled_frames = [keyframes[i] for i in valid_indices]
    sampled_timestamps = [timestamps[i] for i in valid_indices]
    print(f"Sampled frames: {len(sampled_frames)}")

    for idx, frame in enumerate(sampled_frames):
        output_file = os.path.join(output_frame_dir, f"frame_{idx}.jpg")
        print(f"Saving frame {idx} to {output_file}")
        cv2.imwrite(output_file, frame)

    return sampled_frames, valid_indices, sampled_timestamps

'''
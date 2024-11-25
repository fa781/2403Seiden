import av
import numpy as np
from tqdm import tqdm


class PyAVHelper:
    def __init__(self, video_path):
        self.video_path = video_path

    def load_keyframes(self):
        container = av.open(self.video_path)
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = 'NONKEY'

        frames = []
        timestamps = []
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format='rgb24'))
            timestamps.append(frame.pts)
        
        frames = np.stack(frames)
        return frames, timestamps

    def load_video(self):
        container = av.open(self.video_path)
        stream = container.streams.video[0]

        frames = []
        for frame in tqdm(container.decode(stream), desc="Loading frames"):
            frames.append(frame.to_ndarray(format="rgb24"))

        frames = np.stack(frames)
        return frames

    def get_iframe_indices(self):
        container = av.open(self.video_path)
        stream = container.streams.video[0]

        key_indexes = []
        for packet in container.demux(stream):
            if packet.is_keyframe:
                key_indexes.append(packet.pts)
        return np.array(key_indexes)

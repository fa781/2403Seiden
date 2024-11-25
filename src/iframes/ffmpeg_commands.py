import os
import subprocess
import json
import numpy as np


class FFmpegHelper:
    def __init__(self, video_path):
        self.video_path = video_path

    def get_iframe_indices(self):
        command = f"ffprobe -select_streams v -show_frames -show_entries frame=pict_type -of csv {self.video_path}"
        print(f"Command: {command}")
        args = command.split(" ")
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            print(f"Return code is {p.returncode}")
            raise ValueError
        final_output = out.decode('utf-8').split('\n')
        indices = [i for i, x in enumerate(final_output) if x == 'frame,I']
        return np.array(indices)

    def get_video_resolution(self):
        ffprobe_command = f"ffprobe -v error -show_entries stream=width,height -of json {self.video_path}"
        args = ffprobe_command.split(" ")
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            print(f"Return code is {p.returncode}")
            raise ValueError
        dimensions = json.loads(out.decode("utf-8"))["streams"][0]
        return dimensions["width"], dimensions["height"]

    def load_video(self):
        width, height = self.get_video_resolution()
        command = f"ffmpeg -i {self.video_path} -vsync 0 -f rawvideo -pix_fmt rgb24 pipe:"
        args = command.split(" ")
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            print(f"Return code is {p.returncode}")
            raise ValueError
        video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
        return video

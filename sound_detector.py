from aclNet.sound_detector_aclnet import sound_detector_aclnet
from yamnet.sound_detector_yamnet import sound_detector_yamnet
from silence_detector.silence_detector import is_silent

import time
import os
import sys

import wave
import pyaudio
import re
from common_util import logger

ESTIMATED_FRAMES_PER_SECOND = 45
ACCEPTED_ACLNET_PATTERN = r"(?i)aclnet|a"
ACCEPTED_YAMNET_PATTERN = r"(?i)yamnet|y"
DELETE_WAV_FILE_ROUND = 10


class sound_detector:
    def __init__(self, detection_update_rate=1, N=5, output_speed=1, duration_for_sound_detection=3, model="aclnet") -> None:
        self.iterations = [0]
        self.N = N

        self.detect_time_duration = detection_update_rate
        self.detect_frame_buffer = duration_for_sound_detection
        self.output_speed = output_speed
        self.model_param = model
        self.warm_up_is_over = False

        self.aclnet_pattern = re.compile(ACCEPTED_ACLNET_PATTERN)
        self.yamnet_pattern = re.compile(ACCEPTED_YAMNET_PATTERN)

    def compile(self):
        if (re.fullmatch(self.yamnet_pattern, self.model_param)):
            self.model = sound_detector_yamnet()
        elif (re.fullmatch(self.aclnet_pattern, self.model_param)):
            self.model = sound_detector_aclnet()
        else:
            logger.exception(
                "'model' parameter for sound_detector class is wrong!")
            sys.eixt(1)
        self.model.compile()
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        self.frames = []

    def generate_sound_file(self):
        sound_file = wave.open(
            f"myrecording{self.iterations[0]}.wav", "wb")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(b''.join(self.frames))
        sound_file.close()

    def run_once(self):
        time_start = time.perf_counter()
        while time.perf_counter() < self.detect_time_duration + time_start:
            data = self.stream.read(1024)
            self.frames.append(data)

        if not self.warm_up_is_over and self.iterations[0] >= self.detect_frame_buffer:
            self.warm_up_is_over = True
        if self.warm_up_is_over:
            self.frames = self.frames[-(ESTIMATED_FRAMES_PER_SECOND *
                                        self.detect_frame_buffer):]

        if not is_silent(self.frames):
            if self.warm_up_is_over:
                self.generate_sound_file()
                if self.iterations[0] % self.output_speed == 0:
                    self.model.run_inference(self.iterations)
        else:
            logger.info("Silence this turn!!!!")

        self.iterations[0] += 1

        if self.iterations[0] % DELETE_WAV_FILE_ROUND == 0:
            self.delete_temp_wav_files()
            self.iterations[0] = 0

    def run_inference_test(self, file_path):
        self.model.run_inference_test(file_path)

    def delete_temp_wav_files(self):
        for i in range(self.iterations[0] + 1):
            if os.path.isfile(f"myrecording{i}.wav"):
                os.remove(f"myrecording{i}.wav")
            i += 1

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()


def main(model="aclnet"):
    a = sound_detector(model=model)
    a.compile()
    try:
        while True:
            a.run_once()
    except:
        KeyboardInterrupt
    a.delete_temp_wav_files()
    a.close()


def test(model):
    wav_file_folder = "D:/Coding_Workspace/ProjectCodeRepo_ATAP/sound_classification/wav_file"
    a = sound_detector(model=model)
    a.compile()
    for root, dirs, files in os.walk(wav_file_folder):
        for idx, file in enumerate(files):
            logger.info(f"file name: {file}")
            file_path = os.path.join(root, file)
            a.run_inference_test(file_path)


if __name__ == "__main__":
    main()
    # test()

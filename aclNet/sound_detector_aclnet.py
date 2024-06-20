import time
import os

import numpy as np
from openvino import Core
import librosa

import functools
import wave
import torch
import pathlib
from common_util import logger

FILE_PATH = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = os.path.join(FILE_PATH, "aclnet-int8/FP32/aclnet-int8.xml")
LABEL_PATH = os.path.join(FILE_PATH, "aclnet_53cl.txt")

DEVICE = "CPU"
INPUT_SAMPLE_RATE = 44100
EXPECTED_SIZE_FOR_ACLNET = 16000
ESTIMATED_FRAMES_PER_SECOND = 45


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        logger.info(
            f"Elapsed time for function \"{func.__name__}\" is {round((time.perf_counter() - start_time) * 1e3, 2)} ms")
        return value
    return wrapper


class sound_detector_aclnet:
    def __init__(self, N=5) -> None:
        self.N = N

    def compile(self):
        core = Core()
        self.model = core.read_model(MODEL_PATH)
        self.compiled_model = core.compile_model(self.model, DEVICE)
        self.input_tensor_name = self.model.inputs[0].get_any_name()

        self.output_tensor = self.compiled_model.outputs[0]
        self.infer_request = self.compiled_model.create_infer_request()

        self.labels = []
        with open(LABEL_PATH, "r") as file:
            self.labels = [line.rstrip() for line in file.readlines()]

    def chunks(self, audio_data, size, num_chunks=1):
        def get_clip(pos, size):
            if pos > audio_data.shape[1]:
                return np.zeros((audio_data.shape[0], size), dtype=audio_data.dtype)
            if pos + size > audio_data.shape[1]:
                clip = np.zeros(
                    (audio_data.shape[0], size), dtype=audio_data.dtype)
                clip[:, :audio_data.shape[1]-pos] = audio_data[:, pos:]
                return clip
            else:
                return audio_data[:, pos:pos+size]
        pos = 0

        while pos < audio_data.shape[1]:
            chunk = np.zeros(
                (num_chunks, audio_data.shape[0], size), dtype=audio_data.dtype)
            for n in range(num_chunks):
                chunk[n, :, :] = get_clip(pos, size)
                pos += size
            yield chunk

    def read_wav(self, file):
        sampwidth_types = {
            1: np.uint8,
            2: np.int16,
            4: np.int32
        }

        with wave.open(file, "rb") as wav:
            params = wav.getparams()
            data = wav.readframes(params.nframes)
            if params.sampwidth in sampwidth_types:
                data = np.frombuffer(
                    data, dtype=sampwidth_types[params.sampwidth])
            else:
                raise RuntimeError("Couldn't process file {}: unsupported sample width {}"
                                   .format(file, params.sampwidth))
            data = np.reshape(data, (params.nframes, params.nchannels))
            data = (data - np.mean(data)) / (np.std(data) + 1e-15)

        return data

    def resample(self, audio, old_sample_rate, new_sample_rate):
        resample_audio = librosa.resample(
            audio, orig_sr=old_sample_rate, target_sr=new_sample_rate)
        return resample_audio

    def run_inference_test(self, audio):
        data = self.read_wav(audio)
        data = data.T
        data = torch.mean(torch.from_numpy(data), dim=0, keepdim=True).numpy()
        data = self.resample(data, INPUT_SAMPLE_RATE,
                             EXPECTED_SIZE_FOR_ACLNET)
        prediction_list = []

        for _, chunks in enumerate(self.chunks(data, EXPECTED_SIZE_FOR_ACLNET)):
            model_accepted_input = np.reshape(
                chunks, (1, 1, 1, EXPECTED_SIZE_FOR_ACLNET))
            output = self.infer_request.infer(
                {self.input_tensor_name: model_accepted_input})[self.output_tensor]
            data = np.array(output[0])
            prediction_list.append(data)

        mean_of_prediction_list = np.mean(prediction_list, axis=0)
        top_N_classes = np.argsort(mean_of_prediction_list)[-self.N:]

        output = f"The main {self.N} sound predictions is: "
        for i in reversed(top_N_classes):
            output += (self.labels[i]) + " | "

        logger.info(output[:-3])

    def run_inference(self, iterations):
        data = self.read_wav(f"myrecording{iterations[0]}.wav")
        data = data.T
        data = self.resample(data, INPUT_SAMPLE_RATE,
                             EXPECTED_SIZE_FOR_ACLNET)
        prediction_list = []

        for _, chunks in enumerate(self.chunks(data, EXPECTED_SIZE_FOR_ACLNET)):
            model_accepted_input = np.reshape(
                chunks, (1, 1, 1, EXPECTED_SIZE_FOR_ACLNET))
            output = self.infer_request.infer(
                {self.input_tensor_name: model_accepted_input})[self.output_tensor]
            data = np.array(output[0])
            prediction_list.append(data)

        mean_of_prediction_list = np.mean(prediction_list, axis=0)
        top_N_classes = np.argsort(mean_of_prediction_list)[-self.N:]

        output = f"The main {self.N} sound predictions is: "
        for i in reversed(top_N_classes):
            output += (self.labels[i]) + " | "

        logger.info(output[:-3])

    def resample(self, audio, old_sample_rate, new_sample_rate):
        resample_audio = librosa.resample(
            audio, orig_sr=old_sample_rate, target_sr=new_sample_rate)
        return resample_audio

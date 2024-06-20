import os

import time
import functools

import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio

import logging as log
import pathlib
from common_util import logger
tf.get_logger().setLevel('ERROR')

FILE_PATH = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = os.path.join(FILE_PATH, "yamnet_model")


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        log.info(
            f"Elapsed time for function \"{func.__name__}\" is {round((time.perf_counter() - start_time) * 1e3, 2)} ms")
        return value
    return wrapper


class sound_detector_yamnet:
    def __init__(self, N=5) -> None:
        self.N = N

    def compile(self):
        self.yamnet_model = hub.load(
            MODEL_PATH)

    @tf.function
    def load_wav_16k_mono(self, filename):
        """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
        file_contents = tf.io.read_file(filename)
        wav, sample_rate = tf.audio.decode_wav(
            file_contents,
            desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
        return wav

    def run_inference_test(self, file_path):
        class_map_path = self.yamnet_model.class_map_path().numpy().decode('utf-8')
        class_names = list(pd.read_csv(class_map_path)['display_name'])

        wav_data = self.load_wav_16k_mono(file_path)
        scores, embeddings, spectrogram = self.yamnet_model(wav_data)
        class_scores = tf.reduce_mean(scores, axis=0)
        top_N_classes = tf.argsort(class_scores)[-self.N:]
        output = f"The main {self.N} sound predictions is: "
        for i in reversed(top_N_classes.numpy()):
            output += (class_names[i]) + " | "

        logger.info(output[:-3])

    def run_inference(self, iterations):
        class_map_path = self.yamnet_model.class_map_path().numpy().decode('utf-8')
        class_names = list(pd.read_csv(class_map_path)['display_name'])

        wav_data = self.load_wav_16k_mono(
            f"myrecording{iterations[0]}.wav")
        scores, embeddings, spectrogram = self.yamnet_model(wav_data)
        class_scores = tf.reduce_mean(scores, axis=0)
        top_N_classes = tf.argsort(class_scores)[-self.N:]
        output = f"The main {self.N} sound predictions is: "
        for i in reversed(top_N_classes.numpy()):
            output += (class_names[i]) + " | "

        logger.info(output[:-3])

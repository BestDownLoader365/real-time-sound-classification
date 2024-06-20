import numpy as np

SILENCE_THRESHOLD = 100
SILENCE_PROPORTION = 0.9
SILENCE_DETECTION_LENGTH = 44032


def is_silent(audio_frame) -> bool:
    audio_data = np.frombuffer(b''.join(audio_frame), dtype=np.int16)
    bool_return = np.sum(np.abs(audio_data[-SILENCE_DETECTION_LENGTH:]) <=
                         SILENCE_THRESHOLD) / float(SILENCE_DETECTION_LENGTH) >= SILENCE_PROPORTION
    return bool_return

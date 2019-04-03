from pathlib import Path
import mimetypes
import torchaudio

# These are valid file extensions for audio files
AUDIO_EXTENSIONS = set(
    k for k, v in mimetypes.types_map.items() if v.startswith('audio/'))


def getFastAiWorkingDirectory(folder):
    '''Returns the standard working directory for fast.ai for a secific dataset'''
    path = Path(Path.home()/'.fastai/data/')/folder
    if path.exists:
        print(f'Working directory: {path}')
    else:
        print('Missing data folder')
    return path


class AudioData:
    '''Struct that holds basic information from audio signal'''

    def __init__(self, sig, sr=16000):
        self.sig = sig.flatten()  # We want single dimension data
        self.sr = sr

    @classmethod
    def load(cls, fileName, **kwargs):
        p = Path(fileName)
        if p.exists():  # TODO: check type before load because on exception the kernel break
            signal, samplerate = torchaudio.load(str(fileName))
            return AudioData(signal, samplerate)
        raise f'File not fund: {fileName}'

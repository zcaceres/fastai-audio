#Internal dependencies
from .audio import *
from .transform import *


#External dependencies
import mimetypes
from fastai.vision import *
import torchaudio
from torchaudio import transforms

#for jupyter Display
from IPython.display import Audio

class AudioItem(ItemBase):
    def __init__(self, data:AudioData, **kwargs):
        self.data = data # Always flatten out to single dimension signal!
        self.kwargs = kwargs

    def __str__(self):
        return 'REPRESENTATION'
        if isinstance(self.data, AudioData): return f'Duration: {self.duration} seconds.'
        else: return f'{type(self.data)}: {self.data.shape}'
    def __len__(self): return self.data.sig.shape[0]
    def _repr_html_(self): return f'{self.__str__()}<br />{self.ipy_audio._repr_html_()}'

    def show(self, title:Optional[str]=None, **kwargs):
        "Show sound on `ax` with `title`, using `cmap` if single-channel, overlaid with optional `y`"
        self.hear(title=title)

    def hear(self, title=None):
        if title is not None: print(title)
        display(self.ipy_audio)

    def apply_tfms(self, tfms):
        for tfm in tfms:
            self.data = tfm(self.data)
        return self

    @property
    def shape(self):
        return self.data.sig.shape

    @property
    def ipy_audio(self):
        return Audio(data=self.data.sig, rate=self.data.sr)

    @property
    def duration(self): return len(self.data.sig)/self.data.sr

class AudioDataBunch(DataBunch):
    def hear_ex(self, rows:int=3, ds_type:DatasetType=DatasetType.Valid, **kwargs):
        batch = self.dl(ds_type).dataset[:rows]
        self.train_ds.hear_xys(batch.x, batch.y, **kwargs)

class AudioList(ItemList):
    _bunch = AudioDataBunch

    # TODO: __REPR__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get(self, i):
        item = self.items[i]
        if isinstance(item, (Path, str)):
            return AudioItem(AudioData.load(str(item)))
        if isinstance(item, (tuple, np.ndarray)): #data,sr
            return AudioItem(AudioData(item[0],item[1]))
        print('Format not supported!', file=sys.stderr)
        raise

    def reconstruct(self, t:Tensor): return Image(t.transpose(1,2))

    def hear_xys(self, xs, ys, **kwargs):
        for x, y in zip(xs, ys): x.hear(title=y, **kwargs)

    # TODO: example with from_folder
    @classmethod
    def from_folder(cls, path:PathOrStr='.', extensions:Collection[str]=None, **kwargs)->ItemList:
        extensions = ifnone(extensions, AUDIO_EXTENSIONS)
        return super().from_folder(path=path, extensions=extensions, **kwargs)

def get_audio_transforms(spectro:bool=False,
                         white_noise:bool=True,
                         modulate_volume:bool=True,
                         random_cutout:bool=True,
                         pad_with_silence:bool=True,
                         pitch_warp:bool=True,
                         down_and_up:bool=True,
                         mx_to_pad:int=1000,
                         xtra_tfms:Optional[Collection[Transform]]=None,
                         **kwargs)->Collection[Transform]:
    "Utility func to easily create a list of audio transforms."
    res = []
    if white_noise: res.append(partial(tfm_add_white_noise, noise_scl=0.005, **kwargs))
    if modulate_volume: res.append(partial(tfm_modulate_volume, lower_gain=.1, upper_gain=1.2, **kwargs))
    if random_cutout: res.append(partial(tfm_random_cutout, pct_to_cut=.15, **kwargs))
    if pad_with_silence: res.append(partial(tfm_pad_with_silence, pct_to_pad=.15, min_to_pad=None, max_to_pad=None, **kwargs))
    if pitch_warp: res.append(partial(tfm_pitch_warp, shift_by_pitch=None, bins_per_octave=12, **kwargs))
    if down_and_up: res.append(partial(tfm_down_and_up, sr_divisor=2, **kwargs))
    res.append(partial(tfm_pad_to_max, mx=mx_to_pad))
    final_transform = tfm_extract_signal
    if spectro: final_transform = tfm_spectro
    res.append(final_transform)
    #       train                   , valid
    return (res + listify(xtra_tfms), [partial(tfm_pad_to_max, mx=mx_to_pad), final_transform])

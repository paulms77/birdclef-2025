# BirdCLEF 2025 — Bird Species Identification from Soundscape Audio

Multi-label classification of bird species from field recordings, built around
an audio-preprocessing pipeline that attacks the central difficulty of the task:
in a recording of the forest, most of the audio is not a bird.

| | |
| --- | --- |
| **Competition** | [BirdCLEF+ 2025](https://www.kaggle.com/competitions/birdclef-2025) |
| **Task** | Multi-label species identification from soundscape audio |
| **Backbone** | EfficientNet-B0 over mel spectrograms |
| **Environment** | Kaggle notebook (GPU) |

---

## Table of contents

- [The core problem](#the-core-problem)
- [Signal isolation](#signal-isolation)
- [Representative clip selection](#representative-clip-selection)
- [Augmentation](#augmentation)
- [Model and training](#model-and-training)
- [Setup](#setup)
- [Usage](#usage)
- [Notes and open questions](#notes-and-open-questions)

---

## The core problem

Training clips are *focal* recordings — someone pointed a microphone at one
calling bird. Test audio is *soundscape* — continuous field recording with
overlapping species, wind, rain, insects and long stretches of nothing.

So the model is scored on a distribution it never trained on, and the gap is
mostly **noise and silence**. Everything below is about closing it.

## Signal isolation

Three approaches were tried, in order, and the findings matter more than the
code.

### 1. Bandpass filtering

A Butterworth bandpass at 2,048–8,192 Hz, on the reasoning that most bird
vocalisation lives in that band:

```python
y_filtered = butter_bandpass_filter(y, lowcut=2048, highcut=8192, fs=fs)
```

**This did not work as hoped.** Bird calls and the ambient noise occupy the
*same* frequency range — wind, rain and insect noise are not conveniently
below or above the birds. Filtering removes as much signal as noise.

Mean noise subtraction and a 25th-percentile amplitude threshold were tried
alongside, with the same conclusion. `noisereduce` was also evaluated.

### 2. Voice activity detection

Since frequency does not separate bird from not-bird, the next attempt
separates on *time* — find the segments where something is happening at all,
using **Silero VAD**:

```python
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
speech_timestamps = get_speech_timestamps(y, model, sampling_rate=fs)
```

Silero is trained on human speech, not birdsong, so it is being used off-label
— but vocalisation of any kind is structured energy against a flat background,
and in practice it flags a usefully large fraction of the calls.

### 3. Silence trimming

The approach that actually earned its place. Non-silent ranges are detected with
a custom amplitude-threshold pass and concatenated:

```python
nonsilent_ranges = detect_nonsilent_custom(y_filtered, sr,
                                           min_silence_len=300,
                                           silence_thresh=-45)

y_non_silent = np.concatenate([
    y_filtered[int(start * sr / 1000):int(end * sr / 1000)]
    for start, end in nonsilent_ranges
])
```

`min_silence_len=300` ms is deliberately generous — birdsong has short gaps
*within* a single call, and a tighter threshold chops one call into fragments.

Audio is played back inline (`IPython.display.Audio`) at each stage. For audio
work this is not optional: a spectrogram can look fine while the filtered
output is clearly destroyed to the ear.

## Representative clip selection

Some species have hundreds of recordings, many of them near-duplicates of the
same individual in the same session. Training on all of them wastes compute and
biases the model toward whichever bird happened to be recorded most.

The fix is to cluster and sample. Each clip is reduced to a mel-spectrogram
feature vector, compressed with PCA, then **k-means** picks representatives near
each cluster centre:

```python
def extract_padded_melspectrogram(file_path, max_len, fs=32000, n_mels=128):
    y, _ = librosa.load(file_path, sr=fs)
    melspec = librosa.feature.melspectrogram(y=y, sr=fs, n_mels=n_mels)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    pad_width = max_len - melspec_db.shape[1]
    if pad_width > 0:
        melspec_db = np.pad(melspec_db, ((0, 0), (0, pad_width)), mode='constant')
    return melspec_db.flatten()

select_representative_cluster_padded(file_paths, k=10, pca_dim=50)
```

Two variants are implemented — **padding** every spectrogram to the longest in
the set, and **trimming** every spectrogram to the shortest. Padding keeps all
the audio but lets clip length dominate the distance metric; trimming
normalises length but discards the tail of long recordings. Both are kept
because which is better depends on how uniform that species' recordings are.

## Augmentation

### Noise injection at controlled SNR

Rather than adding noise at a fixed amplitude, noise is scaled to a target
signal-to-noise ratio, so quiet and loud recordings are degraded comparably:

```python
class GaussianNoiseSNR(AudioTransform):
    def apply(self, y, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)   # 5–40 dB
        a_signal = np.sqrt(y**2).max()
        a_noise = a_signal / (10 ** (snr / 20))
        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise**2).max()
        return (y + white_noise * (1 / a_white) * a_noise).astype(y.dtype)
```

`PinkNoiseSNR` does the same with pink (1/f) noise via `colorednoise`. Pink noise
matters more than white here — natural ambient sound has far more energy at low
frequencies than white noise does, so pink is the better stand-in for real field
conditions.

### Real-world background noise

Environmental recordings are mixed in as external datasets — chainsaw,
chirping birds, crackling fire, rainfall, thunder, wind. These are the actual
contaminants present in soundscape audio, and training against them directly is
more effective than any synthetic approximation.

### Mixup — additive, not interpolated

```python
class Mixup(nn.Module):
    def forward(self, X, Y, weight=None):
        if torch.rand((1,))[0] < self.mixup_prob:
            perm = torch.randperm(X.shape[0])
            if torch.rand((1,))[0] < self.mixup_double:
                X = X + X[perm]                    # two clips
                Y = torch.clamp(Y + Y[perm], 0, 1)
            else:
                perm2 = torch.randperm(X.shape[0])
                X = X + X[perm] + X[perm2]         # three clips
                Y = torch.clamp(Y + Y[perm] + Y[perm2], 0, 1)
        return X, Y
```

This is the key design decision, and it departs from standard mixup. Classic
mixup *interpolates* (`λx₁ + (1-λ)x₂`) with blended soft labels. Here clips are
**summed** and labels are **OR-ed** (`clamp(Y₁ + Y₂, 0, 1)`).

That is because sound physically superimposes. Two birds calling at once produce
the sum of their waveforms, and both species are genuinely present — so the
label should say *both*, not 60% one and 40% the other. Additive mixup with
union labels synthesises exactly the overlapping-species condition that makes
soundscape audio hard, from focal recordings that contain only one bird.

`Mixup2` keeps the conventional Beta-weighted interpolation available alongside.

| Parameter | Value |
| --- | --- |
| `MIX_BETA` | 0.4 |
| `MIXUP_PROB` | 0.5 |
| `MIXUP_DOUBLE` | 0.6 — probability of 2-clip vs 3-clip mix |
| `MIX_BETA2` | 0.7 |
| `MIXUP2_PROB` | 0.3 |

Also used: `pedalboard`, `audiomentations`, `torch-audiomentations` for
time/pitch effects, and `albumentations` on the spectrogram image.

## Model and training

```python
SR          = 32000      # resample to 32 kHz
DURATION    = 10         # second windows

N_FFT       = 2048       # mel spectrogram
WIN_LENGTH  = 2048
HOP_LENGTH  = 384
N_MELS      = 128
FMIN        = 20
FMAX        = 16000
POWER       = 2.0

MODEL_NAME  = 'efficientnet_b0'
IN_CHANNELS = 3
CRITERION   = 'BCEWithLogitsLoss'
OPTIMIZER   = 'Adam'
LR          = 0.03
WEIGHT_DECAY = 1e-6
EPOCHS      = 5
N_FOLD      = 5
BATCH_SIZE  = 128
SEED        = 42

SECONDARY_COEF = 0.5     # weight applied to secondary_labels
```

Audio becomes a 128-mel spectrogram and is treated as a 3-channel image for the
ImageNet-pretrained EfficientNet-B0. `BCEWithLogitsLoss` rather than
cross-entropy, because this is multi-label — several species can be present in
one window.

**`SECONDARY_COEF = 0.5`** handles the dataset's two label columns.
`primary_label` is the species the recording targets; `secondary_labels` lists
others audible in the background. Secondary labels are real but less certain, so
they contribute at half weight instead of being either ignored or trusted fully.

## Setup

```bash
git clone https://github.com/paulms77/birdclef-2025.git
cd birdclef-2025

pip install librosa soundfile pedalboard colorednoise \
            audiomentations torch-audiomentations albumentations \
            torch timm pandas numpy scikit-learn matplotlib tqdm
```

Competition audio is not redistributed here. On Kaggle, attach the
[BirdCLEF 2025 dataset](https://www.kaggle.com/competitions/birdclef-2025/data)
and the background-noise datasets; running elsewhere, download them and adjust
`Config.root_path`:

```python
class Config:
    root_path   = '/kaggle/input/birdclef-2025'
    train_audio = root_path + '/train_audio'
    taxonomy    = root_path + '/taxonomy.csv'

    chainsaw      = glob.glob('/kaggle/input/chainsaw/*.wav')
    chirpingbird  = glob.glob('/kaggle/input/chirpingbird/*.wav')
    cracklingfire = glob.glob('/kaggle/input/cracklingfire/*.wav')
    rainfall      = glob.glob('/kaggle/input/rainfall/*.wav')
    thunder       = glob.glob('/kaggle/input/thunder/*.wav')
    windblow      = glob.glob('/kaggle/input/windblow/*.wav')
```

## Usage

```bash
jupyter notebook birdclef2025_audio_pipeline.ipynb
```

The notebook is ordered as exploration first, then training:

1. **Audio exploration** — load a clip, inspect waveform and spectrogram, listen
2. **Bandpass filtering** — and why it falls short
3. **Silero VAD** — vocalisation segment detection
4. **Silence trimming** — the filter that is kept
5. **MFCC / mel + k-means** — representative clip selection for over-represented species
6. **Augmentation classes** — SNR noise, mixup, audio transforms
7. **Training** — `main(config)` runs the 5-fold loop

A GPU is needed for training; the exploration cells run on CPU.

## Notes and open questions

**Bird calls and ambient noise share a frequency band.** This is the finding
that shaped everything else — it rules out the obvious frequency-domain fix and
forces the work into the time domain (silence trimming, VAD) and into teaching
the model noise robustness through augmentation instead.

**Are `secondary_labels` ordered by audibility?** The labels may have been
entered by hand in the order the annotator noticed them, which would make
position a weak confidence signal — the first secondary label being more
reliably present than the last. Worth testing; currently all secondary labels
get the same `SECONDARY_COEF`.

**Listen to the output at every stage.** Several preprocessing steps that looked
correct on a spectrogram were audibly broken. Inline playback caught them.

## Related

- [Kaggle](https://github.com/paulms77/Kaggle) — other Kaggle competition notebooks
- [Dacon](https://github.com/paulms77/Dacon) — includes a 1st-place COVID-19 cough-audio detection entry

## License

MIT — see [LICENSE](LICENSE).

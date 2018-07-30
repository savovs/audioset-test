# AudioSet Test
## Generate bounding boxes around sounds in a log mel spectrogram and test Google's [AudioSet](https://research.google.com/audioset/) VGGish model 

Forked from: https://github.com/tensorflow/models/tree/master/research/audioset

Running this ([pipenv](https://github.com/pypa/pipenv) required)
```bash
pipenv install
sh download_and_test.sh
```

will test:
1. Generating a 1kHz sine wave
2. Resampling it
3. Generating a spectrogram from it 
4. Producing input examples from it for the model (AudioSet data comes in this shape)
5. Running examples through the model, which will output semantically meaningful, high-level 128-D embedding

The generated embedding is more compact than raw audio and can be used in a shallower downstream model.

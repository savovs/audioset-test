import numpy as np
from matplotlib import pyplot as plt
import soundfile as sf

import vggish_input
import vggish_params

print('\nTesting spectrogram from .wav file\n')

# Generate a 1 kHz sine wave at 44.1 kHz
num_secs = 4
audio_data, sample_rate = sf.read('../data/audio/bird-chirp.wav')

# Produce a batch of log mel spectrogram examples.
input_batch = vggish_input.waveform_to_examples(audio_data, sample_rate)
print('Log Mel Spectrogram example: ', input_batch[0])

shape_text = '\nComparing shapes:\n (seconds, frames, bands)\n {} input,\n {} config.'.format(
    input_batch.shape, (num_secs, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS))

print(shape_text)

# Raise an error if shapes not equal
np.testing.assert_equal(
    input_batch.shape,
    [num_secs, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS])

def plot(data, name='test.png', content_only=False, show=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.pcolormesh(data)

    if content_only:
        fig.patch.set_visible(False)
        plt.axis('off')
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig('../images/{}'.format(name), bbox_inches=extent)

    else:
        plt.title('Log Mel Spectrogram')
        plt.ylabel('Frequency Bands')
        plt.xlabel('Time Frames')
        plt.savefig('../images/{}'.format(name))

    if show:
        plt.show()

plot(np.rot90(input_batch[0]))
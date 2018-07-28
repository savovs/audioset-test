import numpy as np
from matplotlib import pyplot as plt

import vggish_input
import vggish_params


print('\nTesting spectrograms\n')

# Generate a 1 kHz sine wave at 44.1 kHz
num_secs = 3
freq = 1000
sample_rate = 44100
t = np.linspace(0, num_secs, int(num_secs * sample_rate))
x = np.sin(2 * np.pi * freq * t)


# Produce a batch of log mel spectrogram examples.
input_batch = vggish_input.waveform_to_examples(x, sample_rate)
print('Log Mel Spectrogram example: ', input_batch[0])

# Raise an error if shapes not equal
np.testing.assert_equal(
    input_batch.shape,
    [num_secs, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS])

print('\nWith shape: seconds, frames, bands\n',
    input_batch.shape,
    [num_secs, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS])


# fig = plt.figure()
# fig.patch.set_visible(False)

# ax = fig.add_subplot(111)

# plt.axis('off')
plt.pcolormesh(np.rot90(input_batch[0]))

plt.title('Log Mel Spectrogram of a Sine Wave')
plt.ylabel('Frequency Bands')
plt.xlabel('Time Frames')

# extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
# plt.savefig("../images/test.png", bbox_inches=extent)
plt.savefig("../images/test.png")
# plt.show()  

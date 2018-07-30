import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import soundfile as sf

import vggish_input
import vggish_params

PLOT_DECISION_BOUNDARIES = True

num_secs = 4
audio_data, sample_rate = sf.read('../data/audio/bird-chirp.wav')

# Produce a batch of log mel spectrogram examples.
input_batch = vggish_input.waveform_to_examples(audio_data, sample_rate)
print('Log Mel Spectrogram example: ', input_batch[0])

# Clustering
reduced_data = PCA(n_components=2).fit_transform(input_batch[0])
kmeans = KMeans(random_state=0).fit(reduced_data)

# Plot Spectrogram
def plot(data, name='test.png', content_only=False, show=False, save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.pcolormesh(data)

    if content_only:
        fig.patch.set_visible(False)
        plt.axis('off')
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    else:
        plt.title('Log Mel Spectrogram')
        plt.ylabel('Frequency Bands')
        plt.xlabel('Time Frames')
        

    if save:
        if content_only:
            plt.savefig('../images/{}'.format(name), bbox_inches=extent)
        else:
            plt.savefig('../images/{}'.format(name))

    if show:
        plt.show()

# Plot dimensionally reduced spectrogram (PCA analysis) and clustering centroids 
if PLOT_DECISION_BOUNDARIES:
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.Paired,
            aspect='auto', origin='lower', alpha=.3)

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', color='white', markersize=10)

# Plot the centroids as a white X
centroids = kmeans.cluster_centers_

plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidth=5,
            color='black', zorder=10)
plt.title('K-means clustering on PCA-reduced Log Mel Spectrogram\n')
plt.xticks(())
plt.yticks(())


plot(np.rot90(input_batch[0]), content_only=False)
plt.show()

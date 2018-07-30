import numpy as np
import matplotlib.lines as mlines
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import soundfile as sf

import vggish_input
import vggish_params

PLOT_CONTENT_ONLY = True
PLOT_DECISION_BOUNDARIES = False
PLOT_SPECTROGRAM = True

plt.rcParams['figure.dpi'] = 300

def save_figure(name='test.png'):
    if PLOT_CONTENT_ONLY:
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig("../images/{}".format(name), bbox_inches=extent)
    else:
        plt.savefig("../images/{}".format(name))

num_secs = 4
audio_data, sample_rate = sf.read('../data/audio/bird-chirp.wav')

# Produce a batch of log mel spectrogram examples.
input_batch = vggish_input.waveform_to_examples(audio_data, sample_rate)

data = input_batch[0]

# Clustering
reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(random_state=0).fit(reduced_data)
print(kmeans.labels_)

# Plotting
# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1


fig, ax = plt.subplots()
# plt.show()

if PLOT_DECISION_BOUNDARIES:
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    # plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower', alpha=.3)

x_data_mapped = np.interp(reduced_data[:, 0], [x_min, x_max], [0, 1])
y_data_mapped = np.interp(reduced_data[:, 1], [y_min, y_max], [0, 1])

data_colors = kmeans.labels_.astype(float)

plt.scatter(x_data_mapped, y_data_mapped,
        c=data_colors, alpha=0.3)

# Plot the centroids as a white X
centroids = kmeans.cluster_centers_

x_centroids_mapped = np.interp(centroids[:, 0], [x_min, x_max], [0, 1])
y_centroids_mapped = np.interp(centroids[:, 1], [y_min, y_max], [0, 1])
centroids_mapped = np.array([x_centroids_mapped, y_centroids_mapped]).T

plt.scatter(x_centroids_mapped, y_centroids_mapped,
            marker='x', s=169, linewidth=3,
            color='red', zorder=10, alpha=0.5)

save_figure('clustering.png')

print('\nInitial Centroids:', centroids, '\n')
print('\nMapped Centroids [0, 1]:', centroids_mapped, '\n')

if PLOT_SPECTROGRAM:
    ax.imshow(np.rot90(data, 7, (1, 0)), origin='lower', extent=[0, 1, 0, 1])

if PLOT_CONTENT_ONLY:
    fig.patch.set_visible(False)

else:
    plt.title('Clustering on PCA-reduced Log Mel Spectrogram')
    plt.ylabel('Frequency Bands (scaled from {})'.format(data.shape[0]))
    plt.xlabel('Time Frames (scaled from {})'.format(data.shape[1]))

    cross = mlines.Line2D([], [], color='red', marker='x', linestyle='None',
                            markersize=10, label='Centroids')
    white_dot = mlines.Line2D([], [], color='white', marker='o', linestyle='None',
                            markersize=10, label='PCA Reduced Data')

    plt.legend(handles=[cross, white_dot])

save_figure('spectrogram_clustering.png')


# plt.show()

# if DRAW_BOUNDING_BOXES:


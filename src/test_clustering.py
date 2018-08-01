import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches, lines
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import soundfile as sf

import vggish_input
import vggish_params

PLOT_CONTENT_ONLY = True
PLOT_SPECTROGRAM = True
N_CLUSTERS = 5
BOX_SCALE = 1

plt.rcParams['figure.dpi'] = 300


def save_figure(name='test.png'):
    if PLOT_CONTENT_ONLY:
        fig.patch.set_visible(False)
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig("../images/{}".format(name), bbox_inches=extent)
    else:
        plt.savefig("../images/{}".format(name))


audio_data, sample_rate = sf.read('../data/audio/bird-chirp.wav')

# Produce a batch of log mel spectrogram examples.
input_batch = vggish_input.waveform_to_examples(audio_data, sample_rate)
data = input_batch[0]

print('Sound split into spectrogram batches: ', len(input_batch))

# Principal Component Analysis
reduced_data = PCA(n_components=2).fit_transform(data)

# Clustering
kmeans = KMeans(n_clusters=N_CLUSTERS).fit(reduced_data)

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min(), reduced_data[:, 0].max()
y_min, y_max = reduced_data[:, 1].min(), reduced_data[:, 1].max()

x_data_mapped = np.interp(reduced_data[:, 0], [x_min, x_max], [0, 1])
y_data_mapped = np.interp(reduced_data[:, 1], [y_min, y_max], [0, 1])

# Plot Reduced Data Points with color per label
fig, ax = plt.subplots()
data_colors = kmeans.labels_.astype(float)
plt.scatter(x_data_mapped, y_data_mapped,
            c=data_colors, alpha=0.3)

plt.title('Clustering of PCA-reduced Log Mel Spectrogram')

# Plot the centroids as a white X
centroids = kmeans.cluster_centers_

x_centroids_mapped = np.interp(centroids[:, 0], [x_min, x_max], [0, 1])
y_centroids_mapped = np.interp(centroids[:, 1], [y_min, y_max], [0, 1])
centroids_mapped = np.array([x_centroids_mapped, y_centroids_mapped]).T

plt.scatter(x_centroids_mapped, y_centroids_mapped,
            marker='x', s=169, linewidth=3,
            color='red', zorder=10, alpha=0.5)

plt.ylabel('Frequency Bands (scaled from {})'.format(data.shape[0]))
plt.xlabel('Time Frames (scaled from {})'.format(data.shape[1]))
save_figure('clustering.png')

if PLOT_SPECTROGRAM:
    ax.imshow(np.flip(np.rot90(data), 1), origin='lower', extent=[0, 1, 0, 1])

if PLOT_CONTENT_ONLY:
    fig.patch.set_visible(False)

else:
    plt.title('Clustering on PCA-reduced Log Mel Spectrogram')
    plt.ylabel('Frequency Bands (scaled from {})'.format(data.shape[0]))
    plt.xlabel('Time Frames (scaled from {})'.format(data.shape[1]))

    cross = lines.Line2D([], [], color='red', marker='x', linestyle='None',
                         markersize=10, label='Centroids')
    white_dot = lines.Line2D([], [], color='white', marker='o', linestyle='None',
                             markersize=10, label='PCA Reduced Data')

    plt.legend(handles=[cross, white_dot])

save_figure('spectrogram_clustering.png')

# Get box width and height
unique_labels = np.unique(kmeans.labels_)
print('\Labels length: {}, all labels {}\n'.format(len(kmeans.labels_), kmeans.labels_))
print('Length of reduced data: {}\n'.format(len(reduced_data)))
print('Centroids mapped: ', centroids_mapped)

bounding_boxes = []

for unique_label_index, label in enumerate(unique_labels):
    # Get indexes of data with current label
    indexes = []

    for index, number in enumerate(kmeans.labels_):
        if number == label:
            indexes.append(index)

    # Get data with range [0, 1]
    x_data_matching_label = x_data_mapped[indexes]
    y_data_matching_label = y_data_mapped[indexes]

    center_x = centroids_mapped[unique_label_index][0]
    center_y = centroids_mapped[unique_label_index][1]

    # Calculate width and height
    x_min = min(x_data_matching_label)
    x_max = max(x_data_matching_label)
    width = (x_max - x_min) * BOX_SCALE

    y_min = min(y_data_matching_label)
    y_max = max(y_data_matching_label)
    height = (y_max - y_min) * BOX_SCALE

    left = center_x - width / 2
    bottom = center_y - height / 2

    # Add to dict for saving
    bounding_boxes.append({'center_x': center_x, '': center_y, 'height': height, 'width': width})

    # Plot
    rectangle = patches.Rectangle((left, bottom), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rectangle)

save_figure("../images/boxes.png")


for index, centroid in enumerate(centroids_mapped):
    bounding_boxes[index]['center_x'] = centroid[0]
    bounding_boxes[index]['center_y'] = centroid[1]

if PLOT_CONTENT_ONLY:
    fig = plt.figure()
    # fig.patch.set_visible(False)

    ax = fig.add_subplot(111)
    # plt.imshow(image)

    # plt.axis('off')

    # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # plt.show()

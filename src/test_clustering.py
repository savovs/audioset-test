import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches, lines
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import soundfile as sf
import os
import errno

import vggish_input
import vggish_params

PLOT_CONTENT_ONLY = True
PLOT_SPECTROGRAM = True
PLOT_REDUCED_DATA = False
N_CLUSTERS = 3
BOX_SCALE = 1

# Increasing this would exclude quieter sounds
AMPLITUDE_FILTER_BIAS = 0

label_name = 'bird-chirp'
image_out_directory = '../images/{}'.format(label_name)

# Make an image folder with label_name
try:
    os.makedirs(image_out_directory)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

# Plotting helper function
plt.rcParams['figure.dpi'] = 300


def save_figure(file_name='test.png', label_name='audio_file_name', segment_number=0):
    if PLOT_CONTENT_ONLY:
        fig.patch.set_visible(False)
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.savefig("{image_out_directory}/{label_name}_{segment_number}_{file_name}".format(
            image_out_directory=image_out_directory, label_name=label_name, file_name=file_name, segment_number=segment_number), bbox_inches=extent)

    else:
        plt.savefig("{image_out_directory}/{label_name}_{segment_number}_{file_name}".format(
            image_out_directory=image_out_directory, label_name=label_name, file_name=file_name, segment_number=segment_number))


# Read audio file
audio_data, sample_rate = sf.read('../data/audio/bird-chirp.wav')

# Produce a batch of log mel spectrogram examples.
input_batch = vggish_input.waveform_to_examples(audio_data, sample_rate)

# Iterate over spectrograms, generate bounding boxes and plots
for segment_number, spectrogram in enumerate(input_batch):
    # Reshape spectrogram into (x, y, z) which are (frames, freq_bands, amplitude) respectively
    x, y = np.where(spectrogram)

    # Put the amplitude spectrogram in the Z coordinate
    z = np.empty(len(x))
    for index, xy in enumerate(zip(x, y)):
        z[index] = spectrogram[xy]

    # Filter out low amplitude points
    z_mean = np.mean(z)
    indexes_filtered = np.where(z > z_mean + AMPLITUDE_FILTER_BIAS)

    print('Sound split into spectrogram batches: ', len(input_batch))

    # reduced_data = np.array((x, y, z)).T
    reduced_data = np.array(
        (x[indexes_filtered], y[indexes_filtered], z[indexes_filtered])).T

    # Flip spectrogram to draw spectrogram correctly
    spectrogram = np.flip(np.rot90(spectrogram, 1), 0)

    # Clustering
    kmeans = KMeans(n_clusters=N_CLUSTERS).fit(reduced_data)

    # Map spectrogram to range [0, 1]
    x_min, x_max = reduced_data[:, 0].min(), reduced_data[:, 0].max()
    y_min, y_max = reduced_data[:, 1].min(), reduced_data[:, 1].max()

    x_data_mapped = np.interp(reduced_data[:, 0], [x_min, x_max], [0, 1])
    y_data_mapped = np.interp(reduced_data[:, 1], [y_min, y_max], [0, 1])

    # Plot Reduced Data Points with color per label
    fig, ax = plt.subplots()
    data_colors = kmeans.labels_.astype(float)

    if PLOT_REDUCED_DATA:
        plt.scatter(x_data_mapped, y_data_mapped,
                    c=data_colors, alpha=0.3)

    plt.title('Clustering of Log Mel Spectrogram')

    # Plot the centroids as X marks
    centroids = kmeans.cluster_centers_

    print('Centroids: ', centroids)

    x_centroids_mapped = np.interp(centroids[:, 0], [x_min, x_max], [0, 1])
    y_centroids_mapped = np.interp(centroids[:, 1], [y_min, y_max], [0, 1])
    centroids_mapped = np.array([x_centroids_mapped, y_centroids_mapped]).T

    if PLOT_REDUCED_DATA:
        plt.scatter(x_centroids_mapped, y_centroids_mapped,
                    marker='x', s=169, linewidth=3,
                    color='red', zorder=10, alpha=0.5)

        plt.ylabel('Frequency Bands (scaled from {})'.format(spectrogram.shape[0]))
        plt.xlabel('Time Frames (scaled from {})'.format(spectrogram.shape[1]))
        save_figure('clustering.png', segment_number=segment_number)

    if PLOT_SPECTROGRAM:
        ax.imshow(spectrogram, origin='lower', extent=[0, 1, 0, 1])

    if PLOT_CONTENT_ONLY:
        fig.patch.set_visible(False)

    else:
        plt.title('Clustering of Log Mel Spectrogram')
        plt.ylabel('Frequency Bands (scaled from {})'.format(
            spectrogram.shape[0]))
        plt.xlabel('Time Frames (scaled from {})'.format(spectrogram.shape[1]))

        cross = lines.Line2D([], [], color='red', marker='x', linestyle='None',
                             markersize=10, label='Centroids')
        white_dot = lines.Line2D([], [], color='white', marker='o', linestyle='None',
                                 markersize=10, label='Reduced Data')

        plt.legend(handles=[cross, white_dot])

    save_figure('spectrogram_clustering.png', segment_number=segment_number)

    # Get box width and height
    unique_labels = np.unique(kmeans.labels_)
    print('\Labels length: {}, all labels {}\n'.format(
        len(kmeans.labels_), kmeans.labels_))
    print('Centroids mapped: ', centroids_mapped)

    bounding_boxes = []

    for unique_label_index, label in enumerate(unique_labels):
        # Get indexes of spectrogram with current label
        indexes = []

        for index, number in enumerate(kmeans.labels_):
            if number == label:
                indexes.append(index)

        # Get spectrogram with range [0, 1]
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
        bounding_boxes.append(
            {'center_x': center_x, '': center_y, 'height': height, 'width': width})

        # Plot
        rectangle = patches.Rectangle(
            (left, bottom), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rectangle)

    save_figure("boxes.png", segment_number=segment_number)

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

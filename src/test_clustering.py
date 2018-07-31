import numpy as np
import matplotlib.lines as mlines
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import soundfile as sf
import cv2

import vggish_input
import vggish_params

PLOT_CONTENT_ONLY = True
PLOT_SPECTROGRAM = False

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

# Clustering
reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(random_state=0).fit(reduced_data)

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min(), reduced_data[:, 0].max()
y_min, y_max = reduced_data[:, 1].min(), reduced_data[:, 1].max()

x_data_mapped = np.interp(reduced_data[:, 0], [x_min, x_max], [0, 1])
y_data_mapped = np.interp(reduced_data[:, 1], [y_min, y_max], [0, 1])

print('Mapped reduced data: ', x_data_mapped, y_data_mapped)

# Plot Reduced Data Points
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

print('\nInitial Centroids:', centroids, '\n')
print('\nMapped Centroids [0, 1]:', centroids_mapped, '\n')

if PLOT_SPECTROGRAM:
    ax.imshow(np.flip(np.rot90(data), 1), origin='lower', extent=[0, 1, 0, 1])

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

# Get box width and height
unique_labels = np.unique(kmeans.labels_)
print('\Labels length: {}, all labels {}\n'.format(len(kmeans.labels_), kmeans.labels_))
print('Length of reduced data: {}\n'.format(len(reduced_data)))
print('Centroids mapped: ', centroids_mapped)

bounding_boxes = []

for unique_label_index, label in enumerate(unique_labels):
    # Get indexes of data
    indexes = [] 
    
    for index, number in enumerate(kmeans.labels_):
        if number == label:
            indexes.append(index)

    print('Label: {}, indexes: {}'.format(label, indexes))

    x_data_matching_label = x_data_mapped[indexes]
    y_data_matching_label = y_data_mapped[indexes]

    # Calculate width and height
    x_min = min(x_data_matching_label)
    x_max = max(x_data_matching_label)
    width = x_max - x_min

    y_min = min(y_data_matching_label)
    y_max = max(y_data_matching_label)
    height = y_max - y_min

    # cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255,0,0), 2)

    bounding_boxes.append({'height': height, 'width': width})
    print('\nWidth: {}, height: {}\n'.format(label, width, height))


for index, centroid in enumerate(centroids_mapped):
    bounding_boxes[index]['center_x'] = centroid[0]
    bounding_boxes[index]['center_y'] = centroid[1]

label_name = 'chirp'

# Plot bounding boxes
image = cv2.imread('../images/spectrogram_clustering.png')

image_width = image.shape[1]
image_height = image.shape[0]
print('Image shape: ', image_width, image_height)
print('\nBox Dimensions:\nx1, x2, y1, y2\n')

bounding_box_color = (105, 162, 254) if PLOT_SPECTROGRAM else (1, 1, 1)

for box in bounding_boxes:
    # Calculate coordinates using the [0, 1] range
    # then scale to image pixels
    # x1,y1 ------
    # |          |
    # |          |
    # |          |
    # --------x2,y2
    x1 = box['center_x'] - box['width'] / 2
    x2 = box['center_x'] + box['width'] / 2
    y1 = box['center_y'] - box['height'] / 2
    y2 = box['center_y'] + box['height'] / 2
    print('Coordinates Normalised: ', x1, x2, y1, y2)

    coords = [x1 * image_width, x2 * image_width,
              y1 * image_height, y2 * image_height]

    coords = [int(item) for item in coords]
    print('Coordinates: {}\n'.format(coords))

    x1, x2, y1, y2 = coords
    cv2.rectangle(image, (x1, y1), (x2, y2), bounding_box_color, 2)
    cv2.putText(image, label_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, bounding_box_color, 1, cv2.LINE_AA)


if PLOT_CONTENT_ONLY:
    fig = plt.figure()
    # fig.patch.set_visible(False)

    ax = fig.add_subplot(111)
    # plt.imshow(image)

    # plt.axis('off')

    # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.imshow(image)
    # plt.show()
    # plt.imsave("../images/boxes.png", image)

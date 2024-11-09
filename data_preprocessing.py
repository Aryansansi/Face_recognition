import numpy as np

def create_image_pairs(images, labels):
    num_images = images.shape[0]
    pairs = []
    labels_out = []

    for i in range(num_images):
        for j in range(num_images):
            pairs.append([images[i], images[j]])
            labels_out.append(1 if labels[i] == labels[j] else 0)

    return np.array(pairs), np.array(labels_out)

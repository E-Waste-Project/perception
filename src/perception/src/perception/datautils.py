import h5py
import numpy as np
from pathlib import Path
import cv2
import os


def store_single_hdf5(image, image_id, label, hdf5_dir):
    """ Stores a single image to an HDF5 file.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "image", np.shape(image), h5py.h5t.STD_U8BE, data=image
    )
    meta_set = file.create_dataset(
        "meta", np.shape(label), h5py.h5t.STD_U8BE, data=label
    )
    file.close()


def store_many_hdf5(images, labels, hdf5_dir):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)

    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    meta_set = file.create_dataset(
        "meta", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
    )
    file.close()


def read_single_hdf5(image_id, hdf5_dir):
    """ Stores a single image to HDF5.
        Parameters:
        ---------------
        image_id    integer unique ID for image

        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "r+")

    image = np.array(file["/image"]).astype("uint8")
    label = np.array(file["/meta"]).astype("uint8")

    return image, label


if __name__ == "__main__":
    hdf5_dir = Path("/home/abdelrhman/hdf5_data")
    hdf5_dir.mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(hdf5_dir):
        if filename.endswith('.h5'):
            image, label = read_single_hdf5(int(filename[0:-3]), hdf5_dir)
            # draw a rectangle around the region of interest
            for rect in label:
                cv2.rectangle(image, (rect[0][0], rect[0][1]), (rect[1][0], rect[1][1]), (255, 0, 0), 1)
            cv2.imshow("Image", image)
            cv2.waitKey(0)

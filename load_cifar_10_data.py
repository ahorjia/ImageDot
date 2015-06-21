__author__ = 'ali.ghorbani'

import os
from cifar_item import CifarItem

data_batch_files_directory = "/Users/ali.ghorbani/GoogleDrive/MT/ImageDot/cifar-10-batches-py/"

data_batch_file_names = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
    "test_batch"]

batches_meta = "/Users/ali.ghorbani/GoogleDrive/MT/ImageDot/cifar-10-batches-py/batches.meta"


data_batch_full_file_paths = [os.path.join(data_batch_files_directory, file_name) for file_name in data_batch_file_names]

def unpickle(input_file):
    import cPickle
    fo = open(input_file, 'rb')
    dict_image = cPickle.load(fo)
    fo.close()
    return dict_image

def get_metadata():
    import cPickle
    fo = open(batches_meta, 'rb')
    metadata = cPickle.load(fo)
    fo.close()
    return metadata

labels = get_metadata()['label_names']
num_cases_per_batch = get_metadata()['num_cases_per_batch']
num_vis = get_metadata()['num_vis']

def read_data():
    image_entries = []
    for file_path in data_batch_full_file_paths:
        unpickled = unpickle(file_path)
        batch_data = unpickled['data']
        batch_labels = unpickled['labels']
        file_names = unpickled['filenames']

        assert len(batch_data) == len(batch_labels) == len(file_names)

        for i in range(len(batch_data)):
            image_entries.append(CifarItem(batch_labels[i], labels[batch_labels[i]], batch_data[i], file_names[i]))

    return image_entries

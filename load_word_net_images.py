__author__ = 'ali.ghorbani'

original_directory_path = "/Users/ali.ghorbani/GoogleDrive/MT/ImageDot/ImageNet/original"
destination_directory_path = "/Users/ali.ghorbani/GoogleDrive/MT/ImageDot/ImageNet/destination"

def get_images(folder_path):
    from os import listdir
    from os import path
    from os.path import isfile, join
    only_files = [ path.join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path,f)) ]
    return only_files

def resize_images(original_directory, destination_directory, width, height):

    from PIL import Image
    from os import path
    original_files = get_images(original_directory)

    for original_file in original_files:
        if original_file == '':
            continue

        img = Image.open(original_file)

        img = img.resize((width,height))
        (_, file_name) = path.split(original_file)
        img.save(path.join(destination_directory, file_name))

    pass

def read_data():
    from cifar_item import CifarItem
    from PIL import Image
    import numpy as np
    from os import path
    image_entries = []
    BW_counter = 0
    for file_path in get_images(destination_directory_path):
        if file_path.endswith(".DS_Store"):
            continue
        img = Image.open(file_path)
        data = np.asarray(img.getdata())
        data = np.reshape(data, (-1,), order='F')

        if len(data) == 13300: # Count BW images
            print file_path
            BW_counter += 1

        (_, file_name) = path.split(file_path)

        image_entries.append(CifarItem("", "", data, file_name))

    print "Ignored {0} BW images".format(BW_counter)
    return image_entries

def analyze_original_image_sizes(original_directory):

    from PIL import Image
    import numpy as np
    original_files = get_images(original_directory)

    aspect_ratios = []
    for original_file in original_files:
        if original_file == '':
            continue

        img = Image.open(original_file)

        (x, y) = img.size
        aspect_ratios.append(x * 1.0 / y)

        img.close()
    pass

    print "Total # of items", len(aspect_ratios)
    print "Mean", np.mean(aspect_ratios)
    print "Median", np.median(aspect_ratios)
    print "Std", np.std(aspect_ratios)

    from collections import Counter
    b = Counter(aspect_ratios)
    print "Mod", b.most_common(1)


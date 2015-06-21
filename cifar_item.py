__author__ = 'ali.ghorbani'


class CifarItem:
    def __init__(self, numeric_label, label, image, file_name):
        self.numeric_label = numeric_label
        self.label = label
        self.image = image
        self.image_normalized = self.image / 255.0
        self.file_name = file_name

    def __str__(self):
        "Label:{}, FileName:{}".format(self.label, self.file_name)
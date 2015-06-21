__author__ = 'ali.ghorbani'

import unittest

class TestLoadDataMethods(unittest.TestCase):

    def test_load_cifar_10_data(self):
        import load_cifar_10_data as t
        all_data = t.read_data()
        b1 = all_data[3456]
        b2 = all_data[10345]
        self.assertEqual(b1.file_name, 'king_charles_spaniel_s_000191.png')
        self.assertEqual(b2.file_name, 'bufo_marinus_s_001124.png')
        self.assertEqual(b1.label, 'dog')
        self.assertEqual(b2.label, 'frog')

    def test_show_image_cifar(self):
        import numpy as np
        import load_cifar_10_data as t
        from PIL import Image

        all_data = t.read_data()

        images = [1, 7813, 43983, 55886, 56060, 48889, 23549, 33280, 30735, 11789, 13021, 3456]

        new_im = Image.new('RGB', (32 * len(images), 32))

        for image_index in range(len(images)):
            im = np.reshape(all_data[images[image_index]].image, (32, 32, 3), order='F')
            im = Image.fromarray(im)
            new_im.paste(im, (image_index * 32, 0))

        new_im.show()
        self.assertTrue(True)

    def test_show_image_word_net(self):
        import numpy as np
        import load_word_net_images as l
        from PIL import Image

        all_data = l.read_data()

        images = [20, 529, 446, 618, 836, 928, 775, 622, 437, 182, 844, 560]

        new_im = Image.new('RGB', (133 * len(images), 100))

        for image_index in range(len(images)):
            im = np.reshape(all_data[images[image_index]].image, (13300, 3), order='F')
            data2 = list(tuple(pixel) for pixel in im)

            im2 = Image.new("RGB", (133,100))
            im2.putdata(data2)
            new_im.paste(im2, (image_index * 133, 0))

        new_im.show()
        self.assertTrue(True)

    def test_load_show_word_net_image(self):
        file_path = "/Users/ali.ghorbani/GoogleDrive/MT/ImageDot/ImageNet/destination/n01317294_65.JPEG"
        from PIL import Image
        import numpy as np

        img = Image.open(file_path)
        data = img.getdata()

        np_data = np.asarray(data)

        data2 = list(tuple(pixel) for pixel in np_data)

        img2 = Image.new("RGB", img.size)
        img2.putdata(data2)
        img2.show()
        pass

if __name__ == '__main__':
    unittest.main()

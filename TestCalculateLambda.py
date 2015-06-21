__author__ = 'ali.ghorbani'

import unittest

class TestCalculateLambdaMethods(unittest.TestCase):

    def test_lambda_low_k_cifar(self):
        import CalculateLambda as c

        start_index = 1
        end_index = 3456

        print [index for (index, _, _) in c.find_cifar_10_k_closest(start_index, end_index, 10)]

    def test_lambda_low_k_word_net(self):
        import CalculateLambda as c

        start_index = 20
        end_index = 560

        print [index for (index, _, _) in c.find_word_net_k_closest(start_index, end_index, 3)]

    def test_full_lambda_low_k_word_net(self):
        import CalculateLambda as c

        start_index = 1001
        end_index = 990
        dot_count = 4

        self.show_images(start_index, end_index, [index for (index, _, _) in c.find_word_net_k_closest(start_index, end_index, dot_count)])
        pass


    def show_images(self, start, end, middle):
        import numpy as np
        import load_word_net_images as l
        from PIL import Image

        all_data = l.read_data()

        images = [start] + middle + [end]

        new_im = Image.new('RGB', (133 * len(images), 100))

        for image_index in range(len(images)):
            im = np.reshape(all_data[images[image_index]].image, (13300, 3), order='F')
            data2 = list(tuple(pixel) for pixel in im)

            im2 = Image.new("RGB", (133,100))
            im2.putdata(data2)
            new_im.paste(im2, (image_index * 133, 0))

        new_im.show()
        self.assertTrue(True)

        pass

    def test_lambda_count(self):
        import load_cifar_10_data as t
        import CalculateLambda as c
        all_data = t.read_data()

        start_index = 1
        end_index = 3456

        start = all_data[start_index].image_normalized
        end = all_data[end_index].image_normalized

        less_than_zero = 0
        more_than_one = 0
        the_rest = 0

        for i in range(len(all_data)):
            if i == start_index or i == end_index:
                continue

            lam = c.find_condition_value(start, end, all_data[i].image_normalized)

            if lam < 0:
                less_than_zero += 1
            elif lam > 1:
                more_than_one += 1
            else:
                the_rest += 1

        self.assertEqual(less_than_zero, 1)
        self.assertEqual(more_than_one, 292)
        self.assertEqual(the_rest, 59705)
        self.assertEqual(len(all_data), 60000)

if __name__ == '__main__':
    unittest.main()


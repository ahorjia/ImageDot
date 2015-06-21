__author__ = 'ali.ghorbani'

import numpy as np

def find_condition_value(v_a, v_b, v_i):
    sub1 = np.subtract(v_a, v_b)
    sub2 = np.subtract(v_i, v_b)
    nom = np.dot(sub1.T, sub2)
    denom = np.linalg.norm(sub1, 2)
    return nom / np.square(denom)

def find_cifar_10_k_closest(start_index, end_index, k=1):
    import load_cifar_10_data as t
    all_data = t.read_data()
    return find_k_closest(all_data, start_index, end_index, k)

def find_word_net_k_closest(start_index, end_index, k=1):
    import load_word_net_images as l
    all_data = l.read_data()
    return find_k_closest(all_data, start_index, end_index, k)

def find_k_closest(all_data, start_index, end_index, k):
    import CalculateLambda as c

    start = all_data[start_index].image_normalized
    end = all_data[end_index].image_normalized

    result = []

    for i in range(len(all_data)):
        if i == start_index or i == end_index:
            continue

        lam = c.find_condition_value(start, end, all_data[i].image_normalized)

        if lam < 0:
            lam = 0
        elif lam > 1:
            lam = 1

        v_lam = lam * start + (1 - lam) * end
        sub1 = all_data[i].image_normalized - v_lam
        d_i = np.linalg.norm(sub1, 2)
        result.append((i, d_i, lam))

    sorted_x = sorted(result, key=lambda item: item[1])
    sorted_x = sorted_x[:k]

    sorted_x = sorted(sorted_x, key=lambda item: item[2])

    return sorted_x[:k]

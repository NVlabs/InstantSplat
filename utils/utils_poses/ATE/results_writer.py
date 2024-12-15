# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#!/usr/bin/env python2
import os
# import yaml
import numpy as np


def compute_statistics(data_vec):
    stats = dict()
    if len(data_vec) > 0:
        stats['rmse'] = float(
            np.sqrt(np.dot(data_vec, data_vec) / len(data_vec)))
        stats['mean'] = float(np.mean(data_vec))
        stats['median'] = float(np.median(data_vec))
        stats['std'] = float(np.std(data_vec))
        stats['min'] = float(np.min(data_vec))
        stats['max'] = float(np.max(data_vec))
        stats['num_samples'] = int(len(data_vec))
    else:
        stats['rmse'] = 0
        stats['mean'] = 0
        stats['median'] = 0
        stats['std'] = 0
        stats['min'] = 0
        stats['max'] = 0
        stats['num_samples'] = 0

    return stats


# def update_and_save_stats(new_stats, label, yaml_filename):
#     stats = dict()
#     if os.path.exists(yaml_filename):
#         stats = yaml.load(open(yaml_filename, 'r'), Loader=yaml.FullLoader)
#     stats[label] = new_stats
#
#     with open(yaml_filename, 'w') as outfile:
#         outfile.write(yaml.dump(stats, default_flow_style=False))
#
#     return
#
#
# def compute_and_save_statistics(data_vec, label, yaml_filename):
#     new_stats = compute_statistics(data_vec)
#     update_and_save_stats(new_stats, label, yaml_filename)
#
#     return new_stats
#
#
# def write_tex_table(list_values, rows, cols, outfn):
#     '''
#     write list_values[row_idx][col_idx] to a table that is ready to be pasted
#     into latex source
#
#     list_values is a list of row values
#
#     The value should be string of desired format
#     '''
#
#     assert len(rows) >= 1
#     assert len(cols) >= 1
#
#     with open(outfn, 'w') as f:
#         # write header
#         f.write('      &      ')
#         for col_i in cols[:-1]:
#             f.write(col_i + ' & ')
#         f.write(' ' + cols[-1]+'\n')
#
#         # write each row
#         for row_idx, row_i in enumerate(list_values):
#             f.write(rows[row_idx] + ' &     ')
#             row_values = list_values[row_idx]
#             for col_idx in range(len(row_values) - 1):
#                 f.write(row_values[col_idx] + ' & ')
#             f.write(' ' + row_values[-1]+' \n')

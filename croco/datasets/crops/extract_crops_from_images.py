# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
# 
# --------------------------------------------------------
# Extracting crops for pre-training
# --------------------------------------------------------

import os
import argparse
from tqdm import tqdm
from PIL import Image
import functools
from multiprocessing import Pool
import math


def arg_parser():
    parser = argparse.ArgumentParser('Generate cropped image pairs from image crop list')

    parser.add_argument('--crops', type=str, required=True, help='crop file')
    parser.add_argument('--root-dir', type=str, required=True, help='root directory')
    parser.add_argument('--output-dir', type=str, required=True, help='output directory')
    parser.add_argument('--imsize', type=int, default=256, help='size of the crops')
    parser.add_argument('--nthread', type=int, required=True, help='number of simultaneous threads')
    parser.add_argument('--max-subdir-levels', type=int, default=5, help='maximum number of subdirectories')
    parser.add_argument('--ideal-number-pairs-in-dir', type=int, default=500, help='number of pairs stored in a dir')
    return parser


def main(args):
    listing_path = os.path.join(args.output_dir, 'listing.txt')

    print(f'Loading list of crops ... ({args.nthread} threads)')
    crops, num_crops_to_generate = load_crop_file(args.crops)

    print(f'Preparing jobs ({len(crops)} candidate image pairs)...')
    num_levels = min(math.ceil(math.log(num_crops_to_generate, args.ideal_number_pairs_in_dir)), args.max_subdir_levels)
    num_pairs_in_dir = math.ceil(num_crops_to_generate ** (1/num_levels))

    jobs = prepare_jobs(crops, num_levels, num_pairs_in_dir)
    del crops

    os.makedirs(args.output_dir, exist_ok=True)
    mmap = Pool(args.nthread).imap_unordered if args.nthread > 1 else map
    call = functools.partial(save_image_crops, args)

    print(f"Generating cropped images to {args.output_dir} ...")
    with open(listing_path, 'w') as listing:
        listing.write('# pair_path\n')
        for results in tqdm(mmap(call, jobs), total=len(jobs)):
            for path in results:
                listing.write(f'{path}\n')
    print('Finished writing listing to', listing_path)


def load_crop_file(path):
    data = open(path).read().splitlines()
    pairs = []
    num_crops_to_generate = 0
    for line in tqdm(data):
        if line.startswith('#'):
            continue
        line = line.split(', ')
        if len(line) < 8:
            img1, img2, rotation = line
            pairs.append((img1, img2, int(rotation), []))
        else:
            l1, r1, t1, b1, l2, r2, t2, b2 = map(int, line)
            rect1, rect2 = (l1, t1, r1, b1), (l2, t2, r2, b2)
            pairs[-1][-1].append((rect1, rect2))
            num_crops_to_generate += 1
    return pairs, num_crops_to_generate


def prepare_jobs(pairs, num_levels, num_pairs_in_dir):
    jobs = []
    powers = [num_pairs_in_dir**level for level in reversed(range(num_levels))]

    def get_path(idx):
        idx_array = []
        d = idx
        for level in range(num_levels - 1):
            idx_array.append(idx // powers[level])
            idx = idx % powers[level]
        idx_array.append(d)
        return '/'.join(map(lambda x: hex(x)[2:], idx_array))

    idx = 0
    for pair_data in tqdm(pairs):
        img1, img2, rotation, crops = pair_data
        if -60 <= rotation and rotation <= 60:
            rotation = 0  # most likely not a true rotation
        paths = [get_path(idx + k) for k in range(len(crops))]
        idx += len(crops)
        jobs.append(((img1, img2), rotation, crops, paths))
    return jobs


def load_image(path):
    try:
        return Image.open(path).convert('RGB')
    except Exception as e:
        print('skipping', path, e)
        raise OSError()


def save_image_crops(args, data):
    # load images
    img_pair, rot, crops, paths = data
    try:
        img1, img2 = [load_image(os.path.join(args.root_dir, impath)) for impath in img_pair]
    except OSError as e:
        return []

    def area(sz):
        return sz[0] * sz[1]

    tgt_size = (args.imsize, args.imsize)

    def prepare_crop(img, rect, rot=0):
        # actual crop
        img = img.crop(rect)

        # resize to desired size
        interp = Image.Resampling.LANCZOS if area(img.size) > 4*area(tgt_size) else Image.Resampling.BICUBIC
        img = img.resize(tgt_size, resample=interp)

        # rotate the image
        rot90 = (round(rot/90) % 4) * 90
        if rot90 == 90:
            img = img.transpose(Image.Transpose.ROTATE_90)
        elif rot90 == 180:
            img = img.transpose(Image.Transpose.ROTATE_180)
        elif rot90 == 270:
            img = img.transpose(Image.Transpose.ROTATE_270)
        return img

    results = []
    for (rect1, rect2), path in zip(crops, paths):
        crop1 = prepare_crop(img1, rect1)
        crop2 = prepare_crop(img2, rect2, rot)

        fullpath1 = os.path.join(args.output_dir,  path+'_1.jpg')
        fullpath2 = os.path.join(args.output_dir,  path+'_2.jpg')
        os.makedirs(os.path.dirname(fullpath1), exist_ok=True)

        assert not os.path.isfile(fullpath1), fullpath1
        assert not os.path.isfile(fullpath2), fullpath2
        crop1.save(fullpath1)
        crop2.save(fullpath2)
        results.append(path)

    return results


if __name__ == '__main__':
    args = arg_parser().parse_args()
    main(args)


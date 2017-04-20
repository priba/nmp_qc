#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
    download.py: Download the needed datasets.
"""

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"

import argparse
import os

# Parse optios for downloading
parser = argparse.ArgumentParser(description='Download dataset for Message Passing Algorithm.')
parser.add_argument('datasets', metavar='D', type=str.lower, nargs='+', choices=['qm9'],
                           help='Name of dataset to download [QM9]')
parser.add_argument('-p', '--path', metavar='dir', type=str, nargs=1,
        help='path to store the data (default ./)')

# Download data from url
def download_figshare(url, dir_path):
    url = 'https://api.figshare.com/v2/{endpoint}'


# Download QM9 dataset
def download_qm9(data_dir):
    if os.path.exists(os.path.join(data_dir, 'qm9')):
        print('Found QM9 dataset - SKIP!')
        return
    
    url = 'https://ndownloader.figshare.com/files/'

    # README
    download_data(url + '3195392', data_dir)
    # atomref
    download_data(url + '3195395', data_dir)
    # Validation
    download_data(url + '3195401', data_dir)
    # Uncharacterized
    download_data(url + '3195404', data_dir)
    # dsgdb9nsd.xyz.tar.bz2
    download_data(url + '3195398', data_dir)
    # dsC7O2H10nsd.xyz.tar.bz2
    download_data(url + '3195389', data_dir)

# If not exists creates the specified folder
def prepare_data_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.path is None:
        args.path = './'

    prepare_data_dir(args.path)

    if 'qm9' in args.datasets:
        download_qm9(args.path)
    if 'mutag' in args.datasets:
        download_mutag(args.path)
    if 'enzymes' in args.datasets:
        download_enzymes(args.path)

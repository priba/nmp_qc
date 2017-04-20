#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    download.py: Download the needed datasets.
"""

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"

import argparse
import os
import wget
import zipfile
import tarfile

# Download data from url
def download_figshare(file_name, file_ext, dir_path='./'):
    prepare_data_dir(dir_path)
    url = 'https://ndownloader.figshare.com/files/' + file_name
    wget.download(url, out=dir_path)
    if file_ext == '.zip':
        zip_path = os.path.join(dir_path, file_name)
        zip_ref = zipfile.ZipFile(zip_path,'r')
        zip_ref.extractall(dir_path)
        zip_ref.close()
        os.remove(zip_path)
    elif file_ext == '.tar.bz2':
        tar_path = os.path.join(dir_path, file_name)
        tar_ref = tarfile.open(tar_path,'r:bz2')
        tar_ref.extractall(dir_path)
        os.remove(tar_path)

# Download QM9 dataset
def download_qm9(data_dir):
    data_dir = os.path.join(data_dir, 'qm9')
    if os.path.exists(data_dir):
        print('Found QM9 dataset - SKIP!')
        return
    
    prepare_data_dir(data_dir)

    # README
    download_figshare('3195392', '.txt', data_dir)
    # atomref
    download_figshare('3195395', '.txt', data_dir)
    # Validation
    download_figshare('3195401', '.txt', data_dir)
    # Uncharacterized
    download_figshare('3195404', '.txt', data_dir)
    # dsgdb9nsd.xyz.tar.bz2
    download_figshare('3195398', '.tar.bz2', data_dir)
    # dsC7O2H10nsd.xyz.tar.bz2
    download_figshare('3195389', '.tar.bz2', data_dir)

# Download MUTAG and ENZYMES
def download_mutag_enzymes():
    wget.download(url, out=filename)

# If not exists creates the specified folder
def prepare_data_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == '__main__':
    
    # Parse optios for downloading
    parser = argparse.ArgumentParser(description='Download dataset for Message Passing Algorithm.')
    # Positional arguments
    parser.add_argument('datasets', metavar='D', type=str.lower, nargs='+', choices=['qm9','mutag',
                        'enzymes'], help='Name of dataset to download [QM9,MUTAG,ENZYMES]')
    # I/O
    parser.add_argument('-p', '--path', metavar='dir', type=str, nargs=1,
                        help='path to store the data (default ./)')

    args = parser.parse_args()

    # Check parameters
    if args.path is None:
        args.path = './'
    else:
        args.path = args.path[0]

    # Init folder
    prepare_data_dir(args.path)

    # Select datasets
    if 'qm9' in args.datasets:
        download_qm9(args.path)
    if 'mutag' in args.datasets:
        download_figshare('3132449', '.zip', args.path)
    if 'enzymes' in args.datasets:
        download_figshare('3132446', '.zip', args.path)

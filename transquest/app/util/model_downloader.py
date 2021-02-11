from __future__ import print_function

import logging
import warnings
import zipfile
from os import makedirs
from os.path import dirname
from os.path import exists
from sys import stdout

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class GoogleDriveDownloader:
    """
    Minimal class to download shared files from Google Drive.
    """
    MODEL_SIZE = 3.8
    CHUNK_SIZE = 32768
    DOWNLOAD_URL = 'https://docs.google.com/uc?export=download'

    @staticmethod
    def download_file_from_google_drive(file_id, dest_path, overwrite=False, unzip=False, showsize=False, size=MODEL_SIZE):
        """
        Downloads a shared file from google drive into a given folder.
        Optionally unzips it.

        Parameters
        ----------
        file_id: str
            the file identifier.
            You can obtain it from the sharable link.
        dest_path: str
            the destination where to save the downloaded file.
            Must be a path (for example: './downloaded_file.txt')
        overwrite: bool
            optional, if True forces re-download and overwrite.
        unzip: bool
            optional, if True unzips a file.
            If the file is not a zip file, ignores it.
        showsize: bool
            optional, if True print the current download size.
        size:float
            optional, if given it shows the progress of the download
        Returns
        -------
        None
        """

        destination_directory = dirname(dest_path)
        if not exists(destination_directory):
            makedirs(destination_directory)

        if not exists(dest_path) or overwrite:

            session = requests.Session()

            logger.info('   Downloading {} into {}... '.format(file_id, dest_path))
            stdout.flush()

            response = session.get(GoogleDriveDownloader.DOWNLOAD_URL, params={'id': file_id}, stream=True)

            token = GoogleDriveDownloader._get_confirm_token(response)
            if token:
                params = {'id': file_id, 'confirm': token}
                response = session.get(GoogleDriveDownloader.DOWNLOAD_URL, params=params, stream=True)
                print(response)

            # if showsize:
            #     logger.info("\n")  # Skip to the next line

            current_download_size = [0]

            GoogleDriveDownloader._save_response_content(response, dest_path, showsize, current_download_size, size)
            logger.info('Done.')

            if unzip:
                try:
                    logger.info('Unzipping...')
                    stdout.flush()
                    with zipfile.ZipFile(dest_path, 'r') as z:
                        z.extractall(destination_directory)
                    logger.info('Done.')
                except zipfile.BadZipfile:
                    warnings.warn('Ignoring `unzip` since "{}" does not look like a valid zip file'.format(file_id))

    @staticmethod
    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    @staticmethod
    def _save_response_content(response, destination, showsize, current_size, total_size):
        progress_bar = tqdm(total=total_size)
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(GoogleDriveDownloader.CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    if showsize:
                        # print('\r' + str(float(GoogleDriveDownloader.sizeof_fmt(current_size[0]))/total_size), end=' ')
                        # print('\r' + (format(current_size[0]/(1024*1024*1024), '.1f')), end=' ')
                        gib_value = float(format(current_size[0]/(1024*1024*1024), '.1f'))
                        # print('\r' + str(gib_value), end=' ')
                        progress_bar.update(gib_value)
                        # float(format(current_size[0]/(1024*1024*1024), '.2f'))
                        stdout.flush()
                        current_size[0] += GoogleDriveDownloader.CHUNK_SIZE

    # From https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
    @staticmethod
    def sizeof_fmt(num, suffix='B'):
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return '{:.1f} {}{}'.format(num, unit, suffix)
            num /= 1024.0
        return '{:.1f} {}{}'.format(num, 'Yi', suffix)

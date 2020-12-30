from google_drive_downloader import GoogleDriveDownloader as gdd


def download_from_google_drive(file_id, path):
    gdd.download_file_from_google_drive(file_id=file_id,
                                        dest_path=path + "/model.zip",
                                        unzip=True)

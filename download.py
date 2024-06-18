from utils import DatasetDownloader, process_audio_labels
from config import DATA_URL, DATA_PATH,MAJOR_CATEGORIES, META_DATA, FINAL_META_DATA,AUDIO_PATH


if __name__ == "__main__":
    
    # Path to save the downloaded zip file
    download_path = "./data/ESC-50-master.zip"
    
    
    # Create an instance of DatasetDownloader
    downloader = DatasetDownloader(DATA_URL, download_path, DATA_PATH)
    
    # Download and extract the dataset
    downloader.download_and_extract()

    # Process the audio labels
    process_audio_labels(MAJOR_CATEGORIES, META_DATA,AUDIO_PATH, FINAL_META_DATA)
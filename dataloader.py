from pytorch_lightning import LightningDataModule
import os

class CommonVoiceDataModule(LightningDataModule):
    '''
    Common Voice Data Module.
    
    This class helps handling the download, tokenization 
    and prepare the respective dataloaders.
    
    Parameters:
    -----------
        clips_url: Download link from https://commonvoice.mozilla.org/fr/datasets. 
            Be aware that the download link expire each 36 hours. This will allow to download the audio clips
        labels_folder: Folder that contains the audio transcript with phonemes. 
            If you need to replicate, get the labels from https://commonvoice.mozilla.org/fr/datasets a
            nd use the phonemize helper function to process the labels files
    '''
    def __init__(self, clips_url, labels_folder, language_name):
        self.clips_url = clips_url
        self.labels_folder = labels_folder
        self.language_name = language_name
    
    def prepare_data(self):
        if not os.path.isdir(f" data/{self.language_name}"):
            os.mkdir('temp')
            os.system(f"wget -O temp/temp.tar '{self.clips_url}'")
            os.system(f"mkdir data/{self.language_name}")
            os.system(f"tar -zxf temp/temp.tar -C temp")
            os.system(f"mv temp/clips data/{self.language_name}")
            os.system(f"rm -r temp")
            os.system(f"cp -r {self.labels_folder} data/{self.language_name}/labels")
    def setup(self):
        return None
    def train_dataloader(self):
        return DataLoader()

    def val_dataloader(self):
        return DataLoader()

    def test_dataloader(self):
        return DataLoader()
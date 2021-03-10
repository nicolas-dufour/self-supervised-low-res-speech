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
    def __init__(self, clips_url, labels_folder):
        self.clips_url = clips_url
        self.labels_folder = labels_folder
    
    def prepare_data(self):
        os.system()
    def setup(self):
        
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)
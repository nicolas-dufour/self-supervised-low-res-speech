from pytorch_lightning import LightningDataModule
import os
import requests
from tqdm.notebook import tqdm
import tarfile
from phonemize import phonemize_labels

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
    def __init__(self, clips_url, language_name, labels_folder=None):
        self.clips_url = clips_url
        self.labels_folder = labels_folder
        self.language_name = language_name
    
    def prepare_data(self):
        if not os.path.isdir(f" data/{self.language_name}"):
            os.mkdir('temp')
            local_filename = 'temp/temp.tar'
            with requests.get(url, stream=True) as r:
                total_size_in_bytes= int(r.headers.get('content-length', 0))
                block_size = 1024 #1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                r.raise_for_status()
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=block_size): 
                        # If you have chunk encoded response uncomment if
                        # and set chunk_size parameter to None.
                        #if chunk: 
                        progress_bar.update(len(chunk))
                        f.write(chunk)
            progress_bar.close()
            os.mkdir(f"data/{self.language_name}")
            
            with tarfile.open(name='temp/temp.tar') as tar:
                for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                    tar.extract(member=member)
                            
            os.system(f"mv temp/content/cv-corpus-6.1-2020-12-11/fr/clips data/{self.language_name}")
            if self.labels_folder is None:
                os.mkdir(f"data/{self.language_name}/labels")
                
                os.system(f"cp temp/content/cv-corpus-6.1-2020-12-11/fr/train.tsv data/{self.language_name}/labels/")
                os.system(f"cp temp/content/cv-corpus-6.1-2020-12-11/fr/dev.tsv data/{self.language_name}/labels/")
                os.system(f"cp  temp/content/cv-corpus-6.1-2020-12-11/fr/test.csv data/{self.language_name}/labels/")
            else:
                os.system(f"cp -r {self.labels_folder} data/{self.language_name}/labels")
    
            os.system(f"rm -r temp")
            
    def setup(self):
        return None
    def train_dataloader(self):
        return DataLoader()

    def val_dataloader(self):
        return DataLoader()

    def test_dataloader(self):
        return DataLoader()
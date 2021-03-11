from pytorch_lightning import LightningDataModule
import os
import requests
from tqdm.notebook import tqdm
import tarfile
from glob import glob
from phonemize import phonemize_labels
from torch.utils.data import Dataset

import soundfile as sf


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
        
        self.vocab = None
    def prepare_data(self):
        if not os.path.isdir("data"):
            os.mkdir("data")
        if not os.path.isdir(f"data/{self.language_name}"):
            os.mkdir('temp')
            local_filename = 'temp/temp.tar'
            with requests.get(url, stream=True) as r:
                total_size_in_bytes= int(r.headers.get('content-length', 0))
                block_size = 1024 #1 Kibibyte
                print("Downloading Data")
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                r.raise_for_status()
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=block_size):  
                        progress_bar.update(len(chunk))
                        f.write(chunk)
            progress_bar.close()
            os.mkdir(f"data/{self.language_name}")
            
            print("Untaring Data")
            
            with tarfile.open(name='temp/temp.tar') as tar:
                for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers()), unit='files'):
                    tar.extract(member=member
                                
            clips_paths = [y for x in os.walk('temp') for y in glob(os.path.join(x[0], 'clips'))]                
            os.system(f"mv {clips_paths[0]} data/{self.language_name}")
            if self.labels_folder is None:
                os.mkdir(f"data/{self.language_name}/labels")
                
                train_paths = [y for x in os.walk('temp') for y in glob(os.path.join(x[0], 'train.tsv'))]  
                os.system(f"cp {train_paths[0]} data/{self.language_name}/labels/")
                
                dev_paths = [y for x in os.walk('temp') for y in glob(os.path.join(x[0], 'dev.tsv'))]
                os.system(f"cp {dev_paths[0]} data/{self.language_name}/labels/")
                                
                test_paths = [y for x in os.walk('temp') for y in glob(os.path.join(x[0], 'test.tsv'))]
                os.system(f"cp  {test_paths[0]} data/{self.language_name}/labels/")
            else:
                os.system(f"cp -r {self.labels_folder} data/{self.language_name}/labels")
            self.vocab = list(set([char for phonemes in pd.read_csv(train_paths[0], sep='\t')['sentence_phonemes'] for char in phonemes]))
            os.system(f"rm -r temp")
            
    def setup(self):
        return None
    def train_dataloader(self):
        return DataLoader()

    def val_dataloader(self):
        return DataLoader()

    def test_dataloader(self):
        return DataLoader()

class CommonVoiceDataset(Dataset):
    def __init__(self, clips_paths, labels_path):
        self.clips_paths = self.clip_path
        self.labels = pd.read_csv(labels_path, sep='\t')
    def __getitem__(self,idx):
        path = self.labels.iloc[idx]['path']
        
    def __len__(self):
        return len(self.labels)
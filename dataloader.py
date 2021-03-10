from pytorch_lightning import LightningDataModule
import os
import glob
import pandas as pd
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
        clips_url: str
            Download link from https://commonvoice.mozilla.org/fr/datasets. 
            Be aware that the download link expire after some time. This will allow to download the audio clips
        language_name: str
            Language of the data. If phonemize = True, it need to match https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md
        labels_folder: str
            Folder that contains the audio transcript
        phonemize: bool
            If needed, can allow to transform text data to phonemes data.
    Attributes:
    -----------
        vocab: char list
            List containing all the possible phonemes for our dataset.
    '''
    def __init__(self, clips_url, language_name, labels_folder=None, phonemize=False):
        self.clips_url = clips_url
        self.labels_folder = labels_folder
        self.language_name = language_name
        self.phonemize = phonemize

        self.vocab = None
    def prepare_data(self):
        if not os.path.isdir(f"data"):
            os.mkdir('data')
        if not os.path.isdir(f"data/{self.language_name}"):
            os.mkdir('temp')

            local_filename = 'temp/temp.tar'
            with requests.get(self.clips_url, stream=True) as r:
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
            
            with tarfile.open(name='temp/temp.tar') as tar:
                print('Untaring:')
                for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers()), unit='f'):
                    tar.extract(member=member, path='temp')
            clip_path = [y for x in os.walk('./') for y in glob.glob(os.path.join(x[0], 'clips'))]               
            os.system(f"mv {clip_path[0]} data/{self.language_name}")
            if self.labels_folder is None:
                os.mkdir(f"data/{self.language_name}/labels")
                
                train_path = [y for x in os.walk('./temp') for y in glob.glob(os.path.join(x[0], 'train.tsv'))] 
                os.system(f"cp {train_path[0]} data/{self.language_name}/labels/")
                
                
                
                dev_path = [y for x in os.walk('./temp') for y in glob.glob(os.path.join(x[0], 'dev.tsv'))] 
                os.system(f"cp {dev_path[0]} data/{self.language_name}/labels/")
                
                test_path = [y for x in os.walk('./temp') for y in glob.glob(os.path.join(x[0], 'test.tsv'))] 
                os.system(f"cp {test_path[0]} data/{self.language_name}/labels/")
            else:
                os.system(f"cp -r {self.labels_folder} data/{self.language_name}/labels")
            if self.phonemize:
                print('Phonemizing Train set')
                phonemize_labels(
                    f"data/{self.language_name}/labels/train.tsv",
                    'sentence',
                    self.language_name
                )
                print('Phonemizing Dev set')
                phonemize_labels(
                    f"data/{self.language_name}/labels/dev.tsv",
                    'sentence',
                    self.language_name
                )
                print('Phonemizing Test set')
                phonemize_labels(
                    f"data/{self.language_name}/labels/test.tsv",
                    'sentence',
                    self.language_name
                )
            print('Exctracting phoneme vocab')
            self.vocab = list(set([char for sentence in pd.read_csv("data/{self.language_name}/labels/train.tsv",sep='\t')['sentence_phonemes'] for word in re.split('(W)', sentence) for char in word]))
    
            os.system(f"rm -r temp")
            
    def setup(self):
        return None
    def train_dataloader(self):
        return DataLoader()

    def val_dataloader(self):
        return DataLoader()

    def test_dataloader(self):
        return DataLoader()
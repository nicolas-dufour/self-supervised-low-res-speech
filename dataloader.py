import os
import glob
import pandas as pd
import requests
from tqdm.notebook import tqdm
import tarfile
from glob import glob
import soundfile as sf

from phonemize import phonemize_labels

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence



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
    def __init__(self, clips_url, language_name, tokenizer, labels_folder=None, phonemize=False, batch_size=64):
        self.clips_url = clips_url
        self.labels_folder = labels_folder
        self.language_name = language_name
        self.tokenizer = tokenizer
        self.vocab = None
        self.phonemize = phonemize
        self.batch_size = batch_size
    def prepare_data(self):
        '''
        This function download and preprocess the data from common voice
        '''
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
            clip_path = [y for x in os.walk('./') for y in glob(os.path.join(x[0], 'clips'))]               
            os.system(f"mv {clip_path[0]} data/{self.language_name}")
            if self.labels_folder is None:
                os.mkdir(f"data/{self.language_name}/labels")
                
                train_path = [y for x in os.walk('./temp') for y in glob(os.path.join(x[0], 'train.tsv'))] 
                os.system(f"cp {train_path[0]} data/{self.language_name}/labels/")
                
                dev_path = [y for x in os.walk('./temp') for y in glob(os.path.join(x[0], 'dev.tsv'))] 
                os.system(f"cp {dev_path[0]} data/{self.language_name}/labels/")
                
                test_path = [y for x in os.walk('./temp') for y in glob(os.path.join(x[0], 'test.tsv'))] 
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
            self.vocab = list(set([char for sentence in pd.read_csv("data/{self.language_name}/labels/train.tsv",sep='\t')['sentence_phonemes'] for char in sentence]))
            self.tokenizer.add_tokens(self.vocab)
            os.system(f"rm -r temp")
            
    def setup(self):
        '''
        This function create the respective datasets.
        '''
        self.train_set = CommonVoiceDataset(
            self.clips_url,
             f"data/{self.language_name}/labels/train.tsv",
              self.tokenizer
              )
        self.val_set = CommonVoiceDataset(
            self.clips_url,
             f"data/{self.language_name}/labels/dev.tsv",
              self.tokenizer
              )
        self.test_set = CommonVoiceDataset(
            self.clips_url,
             f"data/{self.language_name}/labels/test.tsv",
              self.tokenizer
              )
    def train_dataloader(self):
        '''
        This function create the train dataloaders. 
        We implemented dynamic padding to be as efficient as possible
        '''
        return DataLoader(
            self.train_set,
            batch_size = self.batch_size,
            shuffle = True,
            collate_fn = collate_common_voice_fn,
            num_workers = 8)

    def val_dataloader(self):
        '''
        This function create the validation dataloaders. 
        We implemented dynamic padding to be as efficient as possible
        '''
        return DataLoader(
            self.val_set,
            batch_size = self.batch_size,
            shuffle = True,
            collate_fn = collate_common_voice_fn,
            num_workers = 8)

    def test_dataloader(self):
        '''
        This function create the test dataloaders. 
        We implemented dynamic padding to be as efficient as possible
        '''
        return DataLoader(
            self.test_set,
            batch_size = self.batch_size,
            shuffle = True,
            collate_fn = collate_common_voice_fn,
            num_workers = 8)

class CommonVoiceDataset(Dataset):
    '''
    Dataset for Common Voice. We use Hugging face tokenizer to process the speech data 
    and tokenize the phoneme labels.

    Parameters:
    ----------
        clips_path: str
            Path for the audio clips
        labels_path: str
            Path for the labels tsv
        tokenizer: Tokenizer
            Hugging Face tokenizer
    '''
    def __init__(self, clips_paths, labels_path, tokenizer):
        self.clips_paths = self.clip_path
        self.labels = pd.read_csv(labels_path, sep='\t')
        self.tokenizer = tokenizer
    def __getitem__(self,idx):
        path = self.clip_path+self.labels.iloc[idx]['path']
        speech, _ = sf.read(path)
        label = self.labels.iloc[idx]['sentence_phonemes']
        return self.tokenizer(speech, return_tensors="pt").input_values, self.tokenizer.encode(list(label))
    def __len__(self):
        return len(self.labels)

def collate_common_voice_fn(batch):
    '''
    Collate function that implement dynamic padding.
    '''
    speech_batch = list()
    labels_batch = list()

    for item in batch:
        speech_batch.append(item[0])
        labels_batch.append(item[1])
    
    return(
        pad_sequence(speech_batch, batch_first=True, padding_value=0),
        pad_sequence(labels_batch, batch_first=True, padding_value=0)
    )


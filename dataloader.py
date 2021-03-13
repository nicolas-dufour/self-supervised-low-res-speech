import os
import glob
import pandas as pd
import requests
from tqdm.notebook import tqdm
import tarfile
from glob import glob
import torchaudio

from phonemize import phonemize_labels

import torch
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
        
        labels_folder: str, default None
            Folder that contains the audio transcript
        
        phonemize: bool, default False
            If needed, can allow to transform text data to phonemes data.
        
        batch_size: int, default True
            The batch_size that is given to the dataloader
        
        label_type: str, default phonemes
            Tell us if we want to have a phoneme based labelisation ('phonemes') or text based ('text')
            
    Attributes:
    -----------
        language_name: str
            The language of the dataset
        
        tokenizer: Object
            The tokenizer function that allow to convert from char to tokens and tokens to char
        
        vocab: char list
            List containing all the possible phonemes for our dataset.
        
        vocab_size: int,
            The size of the vocab
        
        batch_size: int,
            Batch size of the created dataloaders
        
        num_workers: int,
            Workers in the dataloader
        
        label_type: str,
            Type of labels we use. 'phonemes' for the phonetize version of the labels 
            and 'text' for the text version.
        
        train_set: Pytorch Dataset,
            The pytorch dataset for the train set
        
        val_set: Pytorch Dataset,
            The pytorch dataset for the val set
        
        test_set: Pytorch Dataset,
            The pytorch dataset for the test set
    '''
    def __init__(self, clips_url, language_name, labels_folder=None, phonemize=False, label_type='phonemes',batch_size=64, num_workers = 1):
        self.__clips_url = clips_url
        self.__labels_folder = labels_folder
        self.__phonemize = phonemize
        
        self.language_name = language_name
        self.tokenizer = None
        self.vocab = None
        self.vocab_size = None
        self.label_type = label_type
        
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def prepare_data(self):
        '''
        This function download and preprocess the data from common voice
        '''
        if not os.path.isdir(f"data"):
            os.mkdir('data')
        if not os.path.isdir(f"data/{self.language_name}"):
            os.mkdir('temp')
            local_filename = 'temp/temp.tar'
            with requests.get(self.__clips_url, stream=True) as r:
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
            if self.__labels_folder is None:
                os.mkdir(f"data/{self.language_name}/labels")
                
                train_path = [y for x in os.walk('./temp') for y in glob(os.path.join(x[0], 'train.tsv'))] 
                os.system(f"cp {train_path[0]} data/{self.language_name}/labels/")
                
                dev_path = [y for x in os.walk('./temp') for y in glob(os.path.join(x[0], 'dev.tsv'))] 
                os.system(f"cp {dev_path[0]} data/{self.language_name}/labels/")
                
                test_path = [y for x in os.walk('./temp') for y in glob(os.path.join(x[0], 'test.tsv'))] 
                os.system(f"cp {test_path[0]} data/{self.language_name}/labels/")
            else:
                os.system(f"cp -r {self.__labels_folder} data/{self.language_name}/labels")
            if self.__phonemize:
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
            os.system(f"rm -r temp")
        print('Extracting phoneme vocab')
        if self.label_type == 'phonemes':
            self.vocab = list(set([char for sentence in pd.read_csv(f"data/{self.language_name}/labels/train.tsv",sep='\t')['sentence_phonemes'] for char in sentence]))
        elif self.label_type == 'text':
            self.vocab = list(set([char for sentence in pd.read_csv(f"data/{self.language_name}/labels/train.tsv",sep='\t')['sentence'] for char in sentence]))
        else:
            raise "Label type not supported"
        self.tokenizer = PhonemeTokenizer(self.vocab)
        self.vocab_size = len(self.vocab)+2
            
    def setup(self):
        '''
        This function create the respective datasets.
        '''
        resample_transform = torchaudio.transforms.Resample(orig_freq = 48000, new_freq =16000)
        if self.label_type == 'phonemes':
            label_col = 'sentence_phonemes'
        elif self.label_type == 'text':
            label_col = 'sentence'
        else:
            raise "Label type not supported"
        self.train_set = CommonVoiceDataset(
            f"data/{self.language_name}/clips/",
            f"data/{self.language_name}/labels/train.tsv",
            self.tokenizer,
            label_col = label_col,
            transform = resample_transform
            )
        self.val_set = CommonVoiceDataset(
            f"data/{self.language_name}/clips/",
            f"data/{self.language_name}/labels/dev.tsv",
            self.tokenizer,
            label_col = label_col,
            transform = resample_transform
            )
        self.test_set = CommonVoiceDataset(
            f"data/{self.language_name}/clips/",
            f"data/{self.language_name}/labels/test.tsv",
            self.tokenizer,
            label_col = label_col,
            transform = resample_transform
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
            num_workers = self.num_workers
            )

    def val_dataloader(self):
        '''
        This function create the validation dataloaders. 
        We implemented dynamic padding to be as efficient as possible
        '''
        return DataLoader(
            self.val_set,
            batch_size = self.batch_size,
            shuffle = False,
            collate_fn = collate_common_voice_fn,
            num_workers = self.num_workers
            )

    def test_dataloader(self):
        '''
        This function create the test dataloaders. 
        We implemented dynamic padding to be as efficient as possible
        '''
        return DataLoader(
            self.test_set,
            batch_size = self.batch_size,
            shuffle = False,
            collate_fn = collate_common_voice_fn,
            num_workers = self.num_workers
            )

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
    def __init__(self, clips_paths, labels_path, tokenizer, label_col ='sentence_phonemes', transform=None):
        self.clips_paths = clips_paths
        self.labels = pd.read_csv(labels_path, sep='\t')
        self.label_col = label_col
        self.tokenizer = tokenizer
        self.transform = transform
    def __getitem__(self,idx):
        path = self.clips_paths+self.labels.iloc[idx]['path']
        speech, freq = torchaudio.load(path)
        label = self.labels.iloc[idx][self.label_col]
        if self.transform:
            speech = self.transform(speech[0])
        else:
            speech = speech[0]

        return speech, torch.LongTensor(self.tokenizer.encode(label))
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
class PhonemeTokenizer:
    '''
    Allows for encoding and decoding from char to token and token to chars
    
    Parameters:
    ----------
        vocab: list of char,
            Contains all the characters we want to consider. 
            We add 2 special characters, <pad> for the padding 
            and <unk> for characters that aren't in vocab
    
    Attributes:
    -----------
        char_to_token_vocab: dict
            Hash list to map char to token value
        
        token_to_char_vocab: dict
            Hash list to map token values to char
            
    Methods:
    --------
        encode: sentence (str) -> tokens (list of int),
            Convert sentence to list of tokens
        
        decode: list_tokens (list of int) -> sentence (str)
            Convert list of tokens to sentence
    '''
    def __init__(self, vocab):
        self.char_to_token_vocab = {**{'<pad>':0,'<unk>':1},**{char:i+2 for i, char in enumerate(vocab)}}
        self.token_to_char_vocab = {**{'0':'<pad>','1':'<unk>'},**{str(i+2):char for i, char in enumerate(vocab)}}
    def encode(self, sentence):
        '''
        Convert sentence to list of tokens
        
        Parameters:
        ----------
            sentence: str
                Sentence we want to tokenize
        Output:
        -------
            tokens: list of int
                Tokenized output
        '''
        tokens = []
        for char in sentence:
            if char in self.char_to_token_vocab:
                tokens.append(self.char_to_token_vocab[char])
            else:
                tokens.append(self.char_to_token_vocab[1])
        return tokens
    def decode(self, list_tokens):
        '''
        Convert list of tokens to sentence
    
        Parameters:
        ----------
            list_tokens: list of int
                List of tokens we want to convert to str
        Output:
        -------
            sentence: str
                Decoded sentence output
        '''
        output_str_list = []
        for token in list_tokens:
            if token!=0:
                if str(token) in self.token_to_char_vocab:
                    output_str_list.append(self.token_to_char_vocab[str(token)])
                else:
                    output_str_list.append('<unk>')
        return ''.join(output_str_list)

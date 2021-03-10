import pandas as pd
from phonemizer import phonemize
 
# You need to have installed sudo apt-get install festival espeak-ng mbrola

def phonemize_df(file_name, column_name):
    data = pd.read_csv(file_name,sep='\t')
    data['sentence_phonemes'] = phonemize(
        data['sentence'],
        language='fr-fr',
        backend = 'espeak',
        njobs = 8
    )
    data.to_csv(file_name, sep='\t')
    
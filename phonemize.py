import pandas as pd
from phonemizer import phonemize
 
# You need to have installed sudo apt-get install festival espeak-ng mbrola

def phonemize_labels(file_name, column_name, language):
    """
    Phonemize function:
    This function allow to convert text to phonemes. You need to be sure to have installed the backend beforehand (sudo apt-get install festival espeak-ng mbrola).
    Parameters:
    -----------
        file_name: str
            Name of the tsv file that contains the sentences we need to phonemize
        column_name: str
            Name of the columns that contains the sentences we want to phonemize
        language: str
            Language of the sentences. See https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md for reference
        
    """
    
    data = pd.read_csv(file_name,sep='\t')
    data['sentence_phonemes'] = phonemize(
        data[column_name],
        language=language,
        backend = 'espeak',
        language_switch = 'remove-flags',
        njobs = 8
    )
    data.to_csv(file_name, sep='\t')
    
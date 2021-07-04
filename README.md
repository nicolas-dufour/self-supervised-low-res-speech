# Self Supervised Low Ressources Speech retrieval

Our project for the "Algorithms for speech and natural language processing (MVA 2021)" class

This project is based on the wav2vec 2.0 model that has been pretrained on massive unsupervised dataset. We then use this pretrain model and perform transfert learning on languages that have not a lot of labeled data. We study the importance of the pretraining language, the impact of the size of the transfert dataset, and the language similarity impact between pretraining and fine tuning

The following shows the Wav2Vec2 architecture

<p align='center'><img src= 'wav2vec2.png'/></p>

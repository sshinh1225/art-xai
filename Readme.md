Pytorch code for the classification part of  ICMR 2019 paper [Context-Aware Embeddings for Automatic Art Analysis](https://arxiv.org/abs/1904.04985). Originates from this github repo: https://github.com/noagarcia/context-art-classification.

Code from the ARTxAI paper (https://ieeexplore.ieee.org/document/10336532) has also been integrated, which originates from this github repo: https://github.com/Fuminides/context-art-classification.

### Setup

1. Download dataset from [here](https://noagarcia.github.io/SemArt/).

2. For the KGM model, download the pre-computed graph embeddings from [here](https://drive.google.com/file/d/1PSXmyqiDcsdTHBUWyoX5b1EMBQLhJxv-/view?usp=sharing), and save the file into the `Data/` directory.

### Train
    
- To train KGM classifier run:
    
    `python main.py --mode train --model kgm --att $attribute --dir_dataset $semart`

Where `$semart` is the path to SemArt dataset and `$attribute` is the classifier type (i.e. `type`, `school`, `time`, or `author`).

### Test

- To test KGM classifier run:
    
    `python main.py --mode test --model kgm --att $attribute --dir_dataset $semart --model_path $model-file`

Where `$semart` is the path to SemArt dataset, `$attribute` is the classifier type (i.e. `type`, `school`, `time`, or `author`), and `$model-file` is the path to the trained model.

Pre-trained models available to download from:
- [KGM Type](https://drive.google.com/file/d/1zLdvyy6gSw3ENAhLin6c1-DvnvJ9_CU9/view?usp=sharing)
- [KGM School](https://drive.google.com/file/d/1h5sYNsINQRC4LtdoyMcRi16Ho6evx00n/view?usp=sharing)
- [KGM Timeframe](https://drive.google.com/file/d/1QTsNQbQmQFUgRiWiYjvuPQrgtzL89Ynj/view?usp=sharing)
- [KGM Author](https://drive.google.com/file/d/1dFThWHACyX08mBk8lz-4sC8XhF-VMYam/view?usp=sharing)

### Results
 
Classification results on SemArt:

| Model        | Type           | School  |    Timeframe    | Author |
| ------------- |:-------------:| -----:|---------:|--------:|
| VGG16 pre-trained | 0.706 | 0.502 | 0.418 | 0.482 |
| ResNet50 pre-trained | 0.726 | 0.557 | 0.456 | 0.500 | 
| ResNet152 pre-trained | 0.740 | 0.540 | 0.454 | 0.489 |
| VGG16 fine-tuned | 0.768 | 0.616 | 0.559 | 0.520 |
| ResNet50 fine-tuned | 0.765 | 0.655 | 0.604 | 0.515 |
| ResNet152 fine-tuned | 0.790 | 0.653 | 0.598 | 0.573 |
| ResNet50+Attributes | 0.785 | 0.667 | 0.599 | 0.561 |
| ResNet50+Captions | 0.799 | 0.649 | 0.598 | 0.607 |
| MTL context-aware | 0.791 | **0.691** | **0.632** | 0.603 |
| KGM context-aware | **0.815** | 0.671 | 0.613 | **0.615** |  


### Citations


```
@InProceedings{Garcia2017Context,
   author    = {Noa Garcia and Benjamin Renoust and Yuta Nakashima},
   title     = {Context-Aware Embeddings for Automatic Art Analysis},
   booktitle = {Proceedings of the ACM International Conference on Multimedia Retrieval},
   year      = {2019},
}
```
```
@InProceedings{Fuminides,
   author    = {Javier Fumanal-Idocin and Andreu Perez and Oscar Cordon and H. Hagras and Humberto Bustince},
   title     = {ArtxAI: Explainable Artificial Intelligence Curates Deep Representation Learning for Artistic Images using Fuzzy Techniques},
   booktitle = {IEEE Transactions on Fuzzy Systems},
   year      = {2023},
}
```


[1]: http://researchdata.aston.ac.uk/380/
[2]: https://github.com/facebookresearch/visdom

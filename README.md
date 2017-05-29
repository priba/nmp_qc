# Neural Message Passing for Quantum Chemistry

Implementation of different models of Neural Networks on graphs as explained in the article proposed by Gilmer *et al.* [1].

## Installation

    $ pip install -r requirements.txt
    $ python main.py
    
## Installation of rdkit

Running any experiment using QM9 dataset needs installing the [rdkit](http://www.rdkit.org/) package, which can be done 
following the instructions available [here](http://www.rdkit.org/docs/Install.html)

## Data

The data used in this project can be downloaded [here](https://github.com/priba/nmp_qc/tree/master/data).

## Bibliography

- [1] Gilmer *et al.*, [Neural Message Passing for Quantum Chemistry](https://arxiv.org/pdf/1704.01212.pdf), arXiv, 2017.
- [2] Duvenaud *et al.*, [Convolutional Networks on Graphs for Learning Molecular Fingerprints](https://arxiv.org/abs/1606.09375), NIPS, 2015.
- [3] Li *et al.*, [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493), ICLR, 2016. 
- [4] Battaglia *et al.*, [Interaction Networks for Learning about Objects](https://arxiv.org/abs/1612.00222), NIPS, 2016.
- [5] Kipf *et al.*, [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907), ICLR, 2017
- [6] Defferrard *et al.*, [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375), NIPS, 2016. 
- [7] Kearnes *et al.*, [Molecular Graph Convolutions: Moving Beyond Fingerprints](https://arxiv.org/abs/1603.00856), JCAMD, 2016. 
- [8] Bruna *et al.*, [Spectral Networks and Locally Connected Networks on Graphs](https://arxiv.org/abs/1312.6203), ICLR, 2014.
 
 ## Cite
 
```
@Article{Gilmer2017,
  author  = {Justin Gilmer and Samuel S. Schoenholz and Patrick F. Riley and Oriol Vinyals and George E. Dahl},
  title   = {Neural Message Passing for Quantum Chemistry},
  journal = {CoRR},
  year    = {2017}
}
```

## Authors

* Pau Riba (@priba) [Webpage](http://www.cvc.uab.es/people/priba/)
* Anjan Dutta (@AnjanDutta) [Webpage](https://sites.google.com/site/2adutta/)

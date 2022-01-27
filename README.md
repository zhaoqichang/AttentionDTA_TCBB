# AttentionDTA_TCBB
 AttentionDTA: drug--target binding affinity prediction by sequence-based deep learning with attention mechanism

This repository contains the source code and the data.

## Setup and dependencies 

Dependencies:
- python 3.6
- pytorch >=1.2
- numpy
- sklearn
- tqdm
- tensorboardX
- prefetch_generator

## Resources:
+ README.md: this file.
+ datasets: The datasets used in paper.
	+ KIBA.txt:  
	+ Metz.txt: 
	+ Davis.txt
	In the directory of data, we now have the original data "./datasets/KIBA.txt" as follows:

	```
	Drug_ID       Protein_ID   Drug_SMILES          Amino_acid_sequence     affinity
	CHEMBL1087421 O00141       COC1=C(C=C2C(=C...   MTVKTEAAKGTLTYSRMRGM... 11.1
	```
+ dataset.py: data process.
+ AttentionDTA_main.py: train and test the model.
+ Hyperparameter_research.py: hyperparameter seach of AttentionDTA
+ model.py: AttentionDTA model architecture
+ Learning_rate_select.py: find the suitable learning rate



# Run:

python HpyerAttentionDTI_main.py

python Learning_rate_select.py

python Hyperparameter_research.py

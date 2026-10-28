---
license: cc-by-nc-sa-4.0
tags:
- criteo
- advertising
pretty_name: criteo-attribution-dataset
size_categories:
- 10M<n<100M
task_categories:
- tabular-classification
---
# Criteo Attribution Modeling for Bidding Dataset

This dataset is released along with the paper:

[Attribution Modeling Increases Efficiency of Bidding in Display Advertising](https://arxiv.org/abs/1707.06409)
Eustache Diemert&ast;, Julien Meynet&ast; (Criteo Research), Damien Lefortier (Facebook), Pierre Galland (Criteo) &ast;authors contributed equally
2017 AdKDD & TargetAd Workshop, in conjunction with The 23rd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2017) 

When using this dataset, please cite the paper with following bibtex(final ACM bibtex coming soon):

```json
@inproceedings{DiemertMeynet2017,
author = {{Diemert Eustache, Meynet Julien} and Galland, Pierre and Lefortier, Damien},
title={Attribution Modeling Increases Efficiency of Bidding in Display Advertising},
publisher = {ACM},
pages={To appear},
booktitle = {Proceedings of the AdKDD and TargetAd Workshop, KDD, Halifax, NS, Canada, August, 14, 2017},
year = {2017}
}
```

We would love to hear from you if use this data or plan to use it. Refer to the Contact section below.


## Content of this dataset
This dataset includes following files:

  * this **README.md** 
  * **criteo\_attribution\_dataset.tsv.gz**: the dataset itself (623M compressed)
  * **Experiments.ipynb**: ipython notebook with code and utilities to reproduce the results in the paper. Can also be used as a starting point for further research on this data. It requires python 3.* and standard scientific libraries such as pandas, numpy and sklearn.


## Data description
This dataset represents a sample of  30 days of Criteo live traffic data.  Each line corresponds to one impression (a banner) that was displayed to a user.
For each banner we have detailed information about the context, if it was clicked, if it led to a conversion and if it led to a conversion that was attributed to Criteo or not. Data has been sub-sampled and anonymized so as not to disclose proprietary elements.

Here is a detailed description of the fields (they are tab-separated in the file):

  * **timestamp**: timestamp of the impression (starting from 0 for the first impression). The dataset is sorted according to timestamp.
  *  **uid** a unique user identifier
  * **campaign** a unique identifier for the campaign
  * **conversion** 1 if there was a conversion in the 30 days after the impression (independently of whether this impression was last click or not)
  * **conversion_timestamp** the timestamp of the conversion or -1 if no conversion was observed
  *	**conversion_id**	a unique identifier for each conversion (so that timelines can be reconstructed if needed). -1 if there was no conversion
  * **attribution** 1 if the conversion was attributed to Criteo, 0 otherwise
  * **click** 1 if the impression was clicked, 0 otherwise
  *	**click_pos** the position of the click before a conversion (0 for first-click)
  * **click_nb** number of clicks. More than 1 if there was several clicks before a conversion
  * **cost** the price paid by Criteo for this display (**disclaimer:** not the real price, only a transformed version of it)
  *	 **cpo** the cost-per-order  in case of attributed conversion (**disclaimer:** not the real price, only a transformed version of it)
  * **time\_since\_last\_click** the time since the last click (in s) for the given impression
  *	 **cat[1-9]** contextual features associated to the display. Can be used to learn the click/conversion models. We do not disclose the meaning of these features but it is not relevant for this study. Each column is a categorical variable. In the experiments, they are mapped to a fixed dimensionality space using the Hashing Trick (see paper for reference).

### Key figures
  * 2,4Gb uncompressed
  * 16.5M impressions
  * 45K conversions
  * 700 campaigns
  
  

## Tasks
This dataset can be used in a large scope of applications related to Real-Time-Bidding, including but not limited to:

  * Attribution modeling: rule based, model based, etc...
  * Conversion modeling in display advertising: the data includes cost and value used for computing Utility metrics.
  * Offline metrics for real-time bidding
  

## Contact
For any question, feel free to contact:
 
 * The authors of the paper directly (emails in the paper)
 * Criteo Research team: http://ailab.criteo.com/contact-us/
 * Criteo Research twitter account: @CriteoResearch
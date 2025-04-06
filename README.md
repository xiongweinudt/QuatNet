# QuatNet
# Based on RippleNet1.1, we plan to modify the number of triplets in each ripple set. As interest propagates further with the ripple set, the user's interest gradually decreases. Therefore, the number of ripple triplets should also decrease. For example, if we modify the number of ripple triplets to 32 for the 1-hop and 16 for the 2-hop, experiments have shown that there is a certain improvement, although it is minimal.
# Currently, QuatNet is a model with a convolutional quaternion version, but the convolutional output channels are fixed.

This repository is a **PyTorch** implementation of QuatNet:
Based on RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems  
Hongwei Wang, Fuzheng Zhang, Jialin Wang, Miao Zhao, Wenjie Li, Xing Xie, Minyi Guo  
The 27th ACM International Conference on Information and Knowledge Management (CIKM 2018)

For the authors' official TensorFlow implementation, see [hwwang55/RippleNet](https://github.com/hwwang55/RippleNet).

![](https://github.com/hwwang55/RippleNet/blob/master/framework.jpg)

RippleNet is a deep end-to-end model that naturally incorporates the knowledge graph into recommender systems.
Ripple Network overcomes the limitations of existing embedding-based and path-based KG-aware recommendation methods by introducing preference propagation, which automatically propagates users' potential preferences and explores their hierarchical interests in the KG.



### Files in the folder

- `data/`
  - `book/`
    - `BX-Book-Ratings.csv`: raw rating file of Book-Crossing dataset;
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
  - `movie/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG; 
    - `kg_part1.txt` and `kg_part2.txt`: knowledge graph file;
    - `ratrings.dat`: raw rating file of MovieLens-1M;
  - `music/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg.txt`: knowledge graph file;
    - `user_artists.dat`: raw rating file of Music-data

- `src/`: implementations of RippleNet.



### Required packages
The code has been tested running under Python 3.6, with the following packages installed (along with their dependencies):
- pytorch >= 1.0
- numpy >= 1.14.5
- sklearn >= 0.19.1


### Running the code
```
$ cd src
$ python preprocess.py --dataset movie (or --dataset book)
$ python main.py --dataset movie (note: use -h to check optional arguments)
```

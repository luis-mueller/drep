# drep
Implementation of: Diversity Regularized Ensemble Pruning by Li et al. 2012. for Urban Land Use and Land Cover Classification on Remote Sensing Data. More specifically for the `2018 IEEE GRSS Data Fusion Contest: Data Fusion Classification Challenge` which can be found at [http://dase.grss-ieee.org/index.php](http://dase.grss-ieee.org/index.php).

## Setup 
You can install the dependencies of this repository with 

```
pip install requirements.txt
```

You can run the unit tests of this repository with 
```
python -m unittest discover
```

I recommend using python-3, the code was not tested for python-2.

## How to
You can run pruning using the `run_pruning.py` and predict a label map that can be uploaded directly to the contest page with the `run_prediction.py` script. Run 

```
python3 run_pruning.py -h
python3 run_prediction.py -h
```

for details on how the scripts work. You can also provide a config file that allows for chaning multiple experiments in `run_pruning.py` for an example see `_config_hyperparams.json`. Run e.g.

```
python3 run_pruning.py --config=_config_hyperparams.json
```
Note that this code just prunes a collection of labelmaps, not predicts those individual label maps.

Then provide `data/labelmaps` as a source folder for the pruning.

## Data Provisioning
When providing individual label maps, the scripts expect to resolve them via an intermediate layer of groups, i.e. if your labelmaps are stored in a folder `labelmaps` provide them, via something like:

```
labelmaps
    group_a
        labelmap_a_1.tif 
        ...
    group_b
        labelmap_b_1.tif 
        ...
```

Further the estimators expect to find a mapping of classes to RGB-values in a file under `data/classes_rbg.txt`. The file should contain the 
R-, G- and B-value of each class in a separate line, like so:

```
0 10 243
12 210 54
...
```

where line `n` corresponds to a class with label `n - 1`.
I recommend the following overall file structure:

```
data 
    classes_rgb.txt
    labelmaps
        group_a
            ...
        group_b
            ...
```

If you are having problems with executing the code or you find bugs or have ideas for improvements, create PR or an issue.

## Acknowledgement
The implementation of DREP and the corresponding pipeline was applied to the Houston dataset (see above) and thus,
__the author would like to thank the National Center for Airborne Laser Mapping and the Hyperspectral Image Analysis Laboratory at the University of Houston for acquiring and providing the data used in this study, and the IEEE GRSS Image Analysis and Data Fusion Technical Committee.__
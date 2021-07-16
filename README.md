# drep
Implementation of: Diversity Regularized Ensemble Pruning by Li et al. 2012. for Urban Land Use and Land Cover Classification on Remote Sensing Data. More specifically for the `2018 IEEE GRSS Data Fusion Contest: Data Fusion Classification Challenge` which can be found at [http://dase.grss-ieee.org/index.php](http://dase.grss-ieee.org/index.php).

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

## Data Provision
Data is provisioned from a folder (e.g. `data`) and read from there automatically.
Before running the code for first time you need to setup the class-labels and clean the data.
When providing a location for the individual label maps, the file reader provided in this repository will also resolve on additional layer of folder structure. A folder structure could look like:

```
data
  2018_IEEE_GRSS_DFC_GT_TR.tif
  trainSamples.png
  classes.txt
  labelmaps
    group_a
    group_b
    ...
```

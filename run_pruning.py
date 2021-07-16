from library import DrepPipeline
from library.reader import LabelMapIO
import argparse
import json
import numpy as np

# Example: python3 run_pruning.py data/labelmaps/ results/pruning/...


def pruning(source, target, logfile, validationImgPath, suffix, config):  # nSamples, testSize, pruningMethod, diversityTradeoff
    # Load estimators only once (performance)
    estimators = DrepPipeline.loadDB(source.rstrip('/') + '/')
    stats = []
    for conf_ in config:
        process = DrepPipeline(estimators)
        
        resultFile = "_".join([conf_["method"], str(
            conf_["samples"]), str(conf_["testsize"]), str(conf_["div"]), suffix])
        targetPath = target.rstrip('/') + '/' + resultFile


        print("Samples: %d" % conf_["samples"])
        process.sample(conf_["samples"], conf_["testsize"]).prune(
        conf_["method"], conf_["div"])
        
        if validationImgPath != '':
            process.externalValidationStats(validationImgPath)
        else: 
            process.stats()
        
        process.saveModel(targetPath)

        if process.lastStats is not None:
            stats.append(process.lastStats)
    if len(stats) > 0:
        np.savetxt('logs/'+logfile+suffix, stats)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Loads a set of classifiers by recursively resolving a given directory once, i.e. you
    can store your classifiers in one hierarchy of sub-folders. The classifiers are interpreted as an ensemble which is then 
    pruned subsequently. The resulting classifiers are stored in a target file""")
    parser.add_argument('source', type=str,
                        help='Path to the model that is subject to pruning')

    parser.add_argument('target', type=str,
                        help='Path to the file where the list of classifiers is to be stored (after pruning).')

    parser.add_argument('--config', type=str, default='$$NOCONFIG$$',
                        help="""Optional config file (json) with a list of parameters to conduct a multitude of experiments at once.""")

    parser.add_argument('--samples', type=int, default=1000,
                        help="""Number of samples, i.e. number of pixels queried from the training patch for pruning 
                        (including test pixels for validation).""")

    parser.add_argument('--test-size', type=float, default=0,
                        help="""Fraction of samples that are used for validation. Note that these samples are not contributing 
                        to the quality, it is just use as a metric to see immediate quality of the experiement. If 0, no testing is done""")

    parser.add_argument('--pruning-method', type=str, default='drep',
                        help="""Pruning method to be used. Supported are 'identity', which does not prune and can be considered as a baseline,
                        'drep', which implements DREP (2012). More baselines are to be added.""")

    parser.add_argument('--diversity-tradeoff', type=float, default=0.95,
                        help="""DREP (2012) hyperparameter controlling the tradeoff between best performing and most diverse set of classifiers""")

    parser.add_argument('--validation', type=str,
                        help="""If provided with a path to a sparse external validation label map, the stats are produced from that instead of of sampled test data.""")

    parser.add_argument('--suffix', type=str,
                        help="""Suffix added to everything to differentiate runs""")


    args = parser.parse_args()

    if args.config == '$$NOCONFIG$$':
        pruning(args.source, args.target, 'last-via-cmd', [
                {"samples": args.samples, "testsize": args.test_size, "method": args.pruning_method, "div": args.diversity_tradeoff}])
    else:
        config = open(args.config, 'r')
        data = json.load(config)
        pruning(args.source, args.target, args.config.rstrip('.json'), args.validation, args.suffix, data)

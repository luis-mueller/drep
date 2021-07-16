from library import DrepPipeline
import argparse


def prediction(source, target):
    sourceFilename = source.split("/")[-1] + '.tif'
    print(sourceFilename)
    process = DrepPipeline.byModel(source)
    process.predictMap(target.rstrip('/') + '/' + sourceFilename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Loads a saved model that was created with run_pruning.py, predicts
    the DASE 2018 Fusion Contest Label map and stores the result in a .tif file that can directly be uploaded.""")
    parser.add_argument('source', type=str,
                        help='Path to your pruning model.')

    parser.add_argument('target', type=str,
                        help='This is where the resulting label map is stored. Only provide a folder, the name will be the same as the sources')

    args = parser.parse_args()
    prediction(args.source, args.target)

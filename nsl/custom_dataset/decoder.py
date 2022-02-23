import argparse, os, shutil, glob

from torch_geometric.datasets import TUDataset

from nsl.subgraph_mining.config import parse_decoder
from nsl.subgraph_matching.config import parse_encoder
from nsl.subgraph_mining.decoder import pattern_growth


def load_dataset(input_dir, output_dir, name):
    necessary_files = [
        'A.txt', 'graph_indicator.txt', 'graph_labels.txt', 'node_labels.txt'
    ]
    optional_files = [
        'edge_labels.txt', 'edge_attributes.txt', 'node_attributes.txt',
        'graph_attributes.txt'
    ]
    found_files = check_files(input_dir, necessary_files, optional_files)
    copy_files(found_files, output_dir, name)
    dataset = TUDataset(root=output_dir, name=name)
    return dataset


def check_files(input_dir, necessary_files, optional_files):
    all_txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    found_necessary_files = []
    found_optional_files = []
    for file in all_txt_files:
        fname = os.path.basename(file)
        if fname in necessary_files:
            found_necessary_files.append(file)
        elif fname in optional_files:
            found_optional_files.append(file)
    if len(found_necessary_files) != len(necessary_files):
        raise AssertionError('Necessary files miss!')
    found_files = []
    found_files.extend(found_necessary_files)
    found_files.extend(found_optional_files)
    return found_files


def copy_files(found_files, output_dir, name):
    output_dir = os.path.join(output_dir, name, 'raw')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for found_file in found_files:
        fname = os.path.basename(found_file)
        fname = name + '_' + fname
        shutil.copy(found_file, os.path.join(output_dir, fname))


def main():
    if not os.path.exists("plots/cluster"):
        os.makedirs("plots/cluster")

    parser = argparse.ArgumentParser(description='Decoder arguments')
    parse_encoder(parser)
    parse_decoder(parser)
    args = parser.parse_args()

    print("Using dataset in {}".format(args.dataset))
    dataset = load_dataset(args.dataset, output_dir="./", name="DS")
    task = 'graph'
    pattern_growth(dataset, task, args)


if __name__ == '__main__':
    main()

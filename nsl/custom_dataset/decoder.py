import argparse, os, shutil, glob

from torch_geometric.datasets import TUDataset

from nsl.subgraph_mining.config import parse_decoder
from nsl.subgraph_matching.config import parse_encoder
from nsl.subgraph_mining.decoder import pattern_growth


def load_dataset(input_dir, output_dir, name):
    """
    Load local dataset for further analyses.
    
    n = total number of nodes
    m = total number of edges
    N = number of graphs

    (1) A.txt (m lines) 
        sparse (block diagonal) adjacency matrix for all graphs,
        each line corresponds to (row, col) resp. (node_id, node_id)

    (2) graph_indicator.txt (n lines)
        column vector of graph identifiers for all nodes of all graphs,
        the value in the i-th line is the graph_id of the node with node_id i

    (3) graph_labels.txt (N lines) 
        class labels for all graphs in the dataset,
        the value in the i-th line is the class label of the graph with graph_id i

    (4) node_labels.txt (n lines)
        column vector of node labels,
        the value in the i-th line corresponds to the node with node_id i

    There are OPTIONAL files if the respective information is available:

    (5) edge_labels.txt (m lines; same size as DS_A_sparse.txt)
        labels for the edges in DS_A_sparse.txt 

    (6) edge_attributes.txt (m lines; same size as DS_A.txt)
        attributes for the edges in DS_A.txt 

    (7) node_attributes.txt (n lines) 
        matrix of node attributes,
        the comma seperated values in the i-th line is the attribute vector of the node with node_id i

    (8) graph_attributes.txt (N lines) 
        regression values for all graphs in the dataset,
        the value in the i-th line is the attribute of the graph with graph_id i

    Parameters
    ----------

    ==TBC==

    """
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

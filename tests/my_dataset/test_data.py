from typing import Optional, Callable
import os.path as osp
from torch_geometric.datasets import TUDataset

 
class LocalDataset(TUDataset):
    def __init__(self, root: str, name: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None, use_node_attr: bool = False, use_edge_attr: bool = False, cleaned: bool = False):
        super().__init__(root, name, transform, pre_transform, pre_filter, use_node_attr, use_edge_attr, cleaned)
    
    def download(self):
        raise RuntimeError('Please put your raw data files into ', osp.join(self.root, self.name, 'raw'))

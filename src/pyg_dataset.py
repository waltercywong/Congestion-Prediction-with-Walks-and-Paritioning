import os
import sys
import torch
import torch.nn
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm
import pickle
import numpy as np

from utils import *

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (one level above)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path to access the models folder
sys.path.append(parent_dir)

class NetlistDataset(Dataset):
    def __init__(self, data_dir, load_pe=True, load_pd=True, num_eigen=10, pl=True, processed=True, load_indices=None, density=False, method_type="original"):
        super().__init__()
        # Set data_dir to the correct path
        self.data_dir = os.path.join(parent_dir, "data", "superblue")
        self.data_lst = []
        self.load_pe = load_pe
        self.load_pd = load_pd
        self.processed = processed
        self.method_type = method_type

        # Determine the processed filename based on the method
        self.processed_filename = self._get_processed_filename()

        all_files = np.array(os.listdir(self.data_dir))
        
        if load_indices is not None:
            load_indices = np.array(load_indices)
            all_files = all_files[load_indices]
            
        for design_fp in tqdm(all_files):
            data_load_fp = os.path.join(self.data_dir, design_fp, self.processed_filename)
            if processed and os.path.exists(data_load_fp):
                data = torch.load(data_load_fp)
                self._print_loading_message()
            else:
                if self.method_type in ["original", "louvain", "means", "balanced"]:
                    data = self._process_design(self.data_dir, design_fp)
                else:
                    raise FileNotFoundError(
                        f"Processed file {self.processed_filename} not found for design {design_fp}. "
                        "This dataset requires preprocessed files."
                    )
            data['design_name'] = design_fp
            self.data_lst.append(data)

    def _get_processed_filename(self):
        """Determine the processed filename based on the partition method."""
        filename_mapping = {
            "original": "pyg_data.pkl",
            "louvain": "pyg_data_comm.pkl",
            "means": "pyg_data_means.pkl",
            "weighted": "pyg_data_ww.pkl",
            "balanced": "pyg_data_balanced_means.pkl",
            "random_walk": "pyg_data_rw.pkl",
            "xgb": "pyg_data_with_valid_xgb.pkl",
            "xgb_lim": "pyg_data_with_valid_limited.pkl"
        }
        return filename_mapping.get(self.method_type, "pyg_data.pkl")

    def _process_design(self, data_dir, design_fp):
        """Process design data for datasets that allow processing."""
        data_load_fp = os.path.join(data_dir, design_fp)

        # Load node features
        file_name = data_load_fp + '/' + 'node_features.pkl'
        with open(file_name, 'rb') as f:
            dictionary = pickle.load(f)
        
        self.design_name = dictionary['design']
        num_instances = dictionary['num_instances']
        num_nets = dictionary['num_nets']
        raw_instance_features = torch.Tensor(dictionary['instance_features'])
        pos_lst = raw_instance_features[:, :2]

        x_min = dictionary['x_min']
        x_max = dictionary['x_max']
        y_min = dictionary['y_min']
        y_max = dictionary['y_max'] 
        min_cell_width = dictionary['min_cell_width']
        max_cell_width = dictionary['max_cell_width']
        min_cell_height = dictionary['min_cell_height']
        max_cell_height = dictionary['max_cell_height']
        
        X = pos_lst[:, 0].flatten()
        Y = pos_lst[:, 1].flatten()
        
        instance_features = raw_instance_features[:, 2:]
        net_features = torch.zeros(num_nets, instance_features.size(1))
        
        # Load bipartite graph
        file_name = data_load_fp + '/' + 'bipartite.pkl'
        with open(file_name, 'rb') as f:
            dictionary = pickle.load(f)
        
        instance_idx = torch.Tensor(dictionary['instance_idx']).unsqueeze(dim=1).long()
        net_idx = torch.Tensor(dictionary['net_idx']) + num_instances
        net_idx = net_idx.unsqueeze(dim=1).long()
        
        edge_attr = torch.Tensor(dictionary['edge_attr']).float().unsqueeze(dim=1).float()
        edge_index = torch.cat((instance_idx, net_idx), dim=1)
        edge_dir = dictionary['edge_dir']
        v_drive_idx = [idx for idx in range(len(edge_dir)) if edge_dir[idx] == 1]
        v_sink_idx = [idx for idx in range(len(edge_dir)) if edge_dir[idx] == 0] 
        edge_index_source_to_net = edge_index[v_drive_idx]
        edge_index_sink_to_net = edge_index[v_sink_idx]
        
        edge_index_source_to_net = torch.transpose(edge_index_source_to_net, 0, 1)
        edge_index_sink_to_net = torch.transpose(edge_index_sink_to_net, 0, 1)
        
        x = instance_features
        
        example = Data()
        example.__num_nodes__ = x.size(0)
        example.x = x

        # Load degrees
        fn = data_load_fp + '/' + 'degree.pkl'
        with open(fn, "rb") as f:
            d = pickle.load(f)

        example.edge_attr = edge_attr[:2]
        example.cell_degrees = torch.tensor(d['cell_degrees'])
        example.net_degrees = torch.tensor(d['net_degrees'])

        example.x = torch.cat([example.x, example.cell_degrees.unsqueeze(dim=1)], dim=1)
        example.x_net = example.net_degrees.unsqueeze(dim=1)

        # Load partitioning
        part_file = self._get_partition_file(data_load_fp)
        with open(part_file, 'rb') as f:
            part_dict = pickle.load(f)
        
        part_id_lst = [part_dict[idx] for idx in range(len(example.x))]
        part_id = torch.LongTensor(part_id_lst)

        example.num_vn = len(torch.unique(part_id))
        top_part_id = torch.Tensor([0 for idx in range(example.num_vn)]).long()
        example.num_top_vn = len(torch.unique(top_part_id))
        example.part_id = part_id
        example.top_part_id = top_part_id

        # Load net demand and capacity
        file_name = data_load_fp + '/' + 'net_demand_capacity.pkl'
        with open(file_name, 'rb') as f:
            net_demand_dictionary = pickle.load(f)
        net_demand = torch.Tensor(net_demand_dictionary['demand'])

        # Load targets
        file_name = data_load_fp + '/' + 'targets.pkl'
        with open(file_name, 'rb') as f:
            node_demand_dictionary = pickle.load(f)
        node_demand = torch.Tensor(node_demand_dictionary['demand'])
        
        # Load HPWL
        fn = data_load_fp + '/' + 'net_hpwl.pkl'
        with open(fn, "rb") as f:
            d_hpwl = pickle.load(f)
        net_hpwl = torch.Tensor(d_hpwl['hpwl']).float()

        # Load positional encodings
        if self.load_pe:
            file_name = data_load_fp + '/' + 'eigen.10.pkl'
            with open(file_name, 'rb') as f:
                dictionary = pickle.load(f)
            evects = torch.Tensor(dictionary['evects'])
            example.x = torch.cat([example.x, evects[:example.x.shape[0]]], dim=1)
            example.x_net = torch.cat([example.x_net, evects[example.x.shape[0]:]], dim=1)

        # Load neighbor features
        if self.load_pd:
            file_name = data_load_fp + '/' + 'node_neighbor_features.pkl'
            with open(file_name, 'rb') as f:
                dictionary = pickle.load(f)
            pd = torch.Tensor(dictionary['pd'])
            neighbor_list = torch.Tensor(dictionary['neighbor'])
            assert pd.size(0) == num_instances
            assert neighbor_list.size(0) == num_instances
            example.x = torch.cat([example.x, pd, neighbor_list], dim=1)

        # Create final data object
        data = Data(
            node_features=example.x, 
            net_features=example.x_net, 
            edge_index_sink_to_net=edge_index_sink_to_net, 
            edge_index_source_to_net=edge_index_source_to_net, 
            node_demand=node_demand, 
            net_demand=net_demand,
            net_hpwl=net_hpwl,
            batch=example.part_id,
            num_vn=example.num_vn,
            pos_lst=pos_lst
        )
        
        # Save processed data
        data_save_fp = os.path.join(data_load_fp, self.processed_filename)
        torch.save(data, data_save_fp)
        
        return data

    def _get_partition_file(self, data_load_fp):
        """Get the partition file based on the partition method."""
        part_file_mapping = {
            "original": "metis_part_dict.pkl",
            "louvain": "community.pkl",
            "means": "feature_approx_physical_part_dict.pkl",
            "balanced": "balanced_feature_approx_physical_part_dict.pkl",
        }
        return os.path.join(data_load_fp, part_file_mapping.get(self.method_type, "metis_part_dict.pkl"))

    def len(self):
        return len(self.data_lst)

    def get(self, idx):
        return self.data_lst[idx]
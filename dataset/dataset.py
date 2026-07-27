import os
import torch

from torch_geometric.data import InMemoryDataset, Dataset
from tqdm import tqdm

class AttentionDataset(InMemoryDataset):
    def __init__(self, root=None, transform=None, pre_transform=None, split='train', force_reload=False, transform_name=""):
        self.root = root
        self.split = split
        self.transform_name = transform_name
        self.hydrate_name = ""
        super().__init__(root, transform=transform, pre_transform=pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'{self.split}_data_{self.transform_name}.pt']

    def process(self):   

        data_list = torch.load(f'{self.root}/raw/data_list.pt', weights_only=False)
        if len(self.split) > 0:
            assert self.split in ['train', 'val', 'test']
            split = set(torch.load(self.root+'/splits.pt', weights_only=False)[self.split])
            data_list = [data for data in data_list if data.data_idx in split]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            
        self.save(data_list, self.processed_paths[0])


    def hydrate(self, hydrate_list, key, on_x=False, nullify_prompt=False, selection_fn=None, hydration_name=""):
        if selection_fn is not None:
            hydrate_lookup = {element['data_index']: selection_fn(element) for element in hydrate_list}
        else:
            hydrate_lookup = {element['data_index']: element[key] for element in hydrate_list}
        transform_backup = self.transform
        self.transform = None
        hydrated = list()
        print('[i] Hydrating dataset')
        for i in tqdm(range(len(self))):
            data = self.get(i)
            idx = data.data_idx.item()
            assert idx in hydrate_lookup, idx
            val = hydrate_lookup[idx].to(torch.float16).cpu()
            if on_x:
                if nullify_prompt:
                    val[:data.prompt_len,:] = 0.0
                data.x = torch.cat((data.x, val), -1)
            else:
                setattr(data, key, val)
            hydrated.append(data)
        self._indices = None
        self._data_list = None
        self.data, self.slices = self.collate(hydrated)
        self.transform = transform_backup
        self.hydrate_name += f"__{hydration_name}"


    def dump_data_list(self):
        save_dir = self.processed_file_names[0][:-3]
        save_dir += self.hydrate_name
        save_dir = os.path.join(self.root, 'processed', save_dir)
        os.makedirs(save_dir, exist_ok=True)
        for i in tqdm(range(len(self))):
            data = self.get(i).clone()
            torch.save(data, os.path.join(save_dir, f'graph_{i}.pt'))
        return save_dir



class OnDiskAttentionDataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=transform, pre_transform=None)
        self.file_names = sorted(
            [f for f in os.listdir(root) if f.endswith('.pt')],
            key=lambda x: int(x.split('_')[-1].split('.')[0]))


    def len(self):
        return len(self.file_names)


    def get(self, idx):
        data = torch.load(os.path.join(self.root, self.file_names[idx]), weights_only=False)
        return data
    
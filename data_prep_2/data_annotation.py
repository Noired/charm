import argparse
import os
import torch
import tqdm
import glob


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./raw_data/att_movies-3k_6h-mistral-7b-i-001_24_28_32_16bits_42.pt")
    parser.add_argument("--anno_path", type=str, default="./raw_data/anno_movies-3k_6h-mistral-7b-i-001_24_28_32_16bits_42.pt")
    parser.add_argument("--dataset_root", type=str, default="./data/movies-3k_6h-mistral-7b-i-001_16bits_42/")
    parser.add_argument("--expected_train", type=int)
    parser.add_argument("--expected_test", type=int)

    args = parser.parse_args()

    def load_from_dir(path):
        print(f'Loading from path {path}')
        data_list = list()
        for data_path in tqdm.tqdm(glob.glob(path)):
            data_list.append(torch.load(data_path, weights_only=False))
        return data_list
    
    print(f'[i] Loading unlabeled data and annotating...')
    print('[i]... train data folder')
    att_dict_list = load_from_dir(args.data_path+'/att*')
    print('[i]... test data folder')
    att_dict_list_test = load_from_dir(args.data_path+'_test/att*')
    print('[i]... train annotations folder')
    anno_dict_list = load_from_dir(args.anno_path+'/anno*')
    print('[i]... test annotations folder')
    anno_dict_list_test = load_from_dir(args.anno_path+'_test/anno*')
    assert len(att_dict_list) == len(anno_dict_list) == args.expected_train, (len(att_dict_list), len(anno_dict_list), args.expected_train)
    assert len(att_dict_list_test) == len(anno_dict_list_test) == args.expected_test, (len(att_dict_list_test), len(anno_dict_list_test), args.expected_test)
    
    target_folder = os.path.join(args.dataset_root, 'raw')
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"[i] Created directory '{target_folder}'.")
    else:
        print(f"[i] Directory '{target_folder}' already exists.")
    
    data_list = list()
    label_dict = {data['data_index']: torch.tensor(data['annotation']).float() for data in anno_dict_list}
    for element in att_dict_list:
        idx = element['data_index']
        data = element['data']
        data.y = label_dict[idx]
        data.data_idx = idx
        data_list.append(data)
    offset = len(data_list)
    for element in anno_dict_list_test:
        assert element['data_index']+offset not in label_dict
        label_dict[element['data_index']+offset] = torch.tensor(element['annotation']).float()
    for element in att_dict_list_test:
        idx = element['data_index']
        data = element['data']
        data.y = label_dict[offset + idx]
        data.data_idx = offset + idx
        data_list.append(data)
    assert len(data_list) == len(label_dict) == args.expected_train + args.expected_test

    torch.save(data_list, os.path.join(target_folder, 'data_list.pt'))
    torch.save(label_dict, os.path.join(target_folder, 'labels.pt'))

    # Consolidate other signals
    def consolidate(prefix, data_path):
        dict_list = load_from_dir(data_path+f'/{prefix}*')
        dict_list_test = load_from_dir(args.data_path+f'_test/{prefix}*')
        offset = len(dict_list)
        for element in dict_list_test:
            element['data_index'] += offset
        dict_list.extend(dict_list_test)
        return dict_list

    print('[i] Consolidating atps and activations into a single file...')
    atp_dict = consolidate('atp', args.data_path)
    if len(atp_dict) > 0:
        assert len(atp_dict) == args.expected_train + args.expected_test, len(atp_dict)
        torch.save(atp_dict, os.path.join(args.dataset_root, 'raw', 'all_atps.pt'))
    act_dict = consolidate('act', args.data_path)
    if len(act_dict) > 0:
        assert len(act_dict) == args.expected_train + args.expected_test, len(act_dict)
        torch.save(act_dict, os.path.join(args.dataset_root, 'raw', 'all_acts.pt'))

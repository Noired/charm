import argparse
import os
import torch
import tqdm
import glob


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--atp_path", type=str, default="./raw_data/atp_movies-3k_6h-mistral-7b-i-001_24_28_32_16bits.pt")
    parser.add_argument("--act_path", type=str, default="./raw_data/act_movies-3k_6h-mistral-7b-i-001_24_28_32_16bits.pt")
    parser.add_argument("--out_pattern", type=str, default="movies-3k_6h-mistral-7b-i-001_24_28_32_16bits")
    parser.add_argument("--expected_train", type=int)
    parser.add_argument("--expected_test", type=int)

    args = parser.parse_args()

    def load_from_dir(path):
        print(f'Loading from path {path}')
        data_list = list()
        for data_path in tqdm.tqdm(glob.glob(path)):
            data_list.append(torch.load(data_path, weights_only=False))
    
    print(f'[i] Loading atp data...')
    if os.path.isdir(args.atp_path):
        print('[i]... train data folder')
        atp_dict_list = load_from_dir(args.atp_path+'/atp*')
        print('[i]... test data folder')
        atp_dict_list_test = load_from_dir(args.atp_path[:-4]+'test_temp/atp*')
    else:
        print('[i]... train data')
        atp_dict_list = torch.load(args.atp_path, weights_only=False)
        print('[i]... test data')
        atp_dict_list_test = torch.load(args.atp_path[:-3]+'_test.pt', weights_only=False)
    
    print(f'[i] Loading act data...')
    if os.path.isdir(args.act_path):
        print('[i]... train data folder')
        act_dict_list = load_from_dir(args.act_path+'/act*')
        print('[i]... test data folder')
        act_dict_list_test = load_from_dir(args.act_path[:-4]+'test_temp/act*')
    else:
        print('[i]... train data')
        act_dict_list = torch.load(args.act_path, weights_only=False)
        print('[i]... test data')
        act_dict_list_test = torch.load(args.act_path[:-3]+'_test.pt', weights_only=False)
    
    offset = len(atp_dict_list)
    for element in atp_dict_list_test:
        element['data_index'] = element['data_index'] + offset
        atp_dict_list.append(element)
    assert len(atp_dict_list) == args.expected_train + args.expected_test, len(atp_dict_list)

    assert len(act_dict_list) == args.expected_train and len(act_dict_list_test) == args.expected_test
    offset = len(act_dict_list)
    for element in act_dict_list_test:
        element['data_index'] = element['data_index'] + offset
        act_dict_list.append(element)
    assert len(act_dict_list) == args.expected_train + args.expected_test, len(act_dict_list)

    torch.save(atp_dict_list, './raw_data/atp_'+args.out_pattern+'_train_test.pt')
    torch.save(act_dict_list, './raw_data/act_'+args.out_pattern+'_train_test.pt')


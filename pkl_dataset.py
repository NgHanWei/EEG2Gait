from torch.utils.data import Dataset
import torch
from einops import rearrange
import pickle
import os
import re
from pathlib import Path
import numpy as np
standard_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'T1', 'T2', 'I1', 'I2', 'HEO', 'VEO', 'ECG', 'EMG', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6', \
    'EMG7', 'EMG8', 'EMG9', 'EMG10', 'EMG11', 'EMG12', \
    'F3-C3', 'F4-C4', 'F7-T3', 'F8-T4', 'FP1-F3', 'P3-O1', 'P4-O2', 'T3-T5', 'T4-T6', \
    'C3-P3', 'C4-A1', 'C4-P4', 'F2-F4', 'FPZ-CZ', 'PZ-OZ', 'FP2-F4', 'F1-F3'
]

def extract_last_number(filename):
    numbers = re.findall(r'\d+', filename)  
    return int(numbers[-1]) if numbers else float('inf') 


def get_chans(ch_names):
    chans = []
    for ch_name in ch_names:
        chans.append(standard_1020.index(ch_name))
    return chans
def extract_session_and_number(path):
    # ".../6_1/3425.pkl"
    path = os.path.normpath(path)
    parts = path.strip().split(os.sep)
    # print(parts)
    session = parts[-2]  # "6_1"
    filename = parts[-1]  # "3425.pkl"
    
    # session
    s_major, s_minor = map(int, session.split('_'))
    
    number = int(os.path.splitext(filename)[0])
    
    return (s_major, s_minor, number)
    
def extract_number_then_session(path):
    # ".../6_1/3425.pkl"
    path = os.path.normpath(path)
    parts = path.strip().split(os.sep)
    # print(parts)
    session = parts[-2]  # "6_1"
    filename = parts[-1]  # "3425.pkl"
    
    # session
    s_major, s_minor = map(int, session.split('_'))
    
    number = int(os.path.splitext(filename)[0])
    return (number, s_major, s_minor)     

class NPYLoader(Dataset):
    def __init__(self, data_base, session2use, start, end):
        self.data_base = os.path.join(data_base, session2use)
        self.start = start
        self.end = end
        self.EEG = []
        self.goni = []
        self.phase = []
        
        for i in range(start, end):
            if os.path.exists(os.path.join(self.data_base, f'EEG_trial_{i}.npy')):
                EEG_file = os.path.join(self.data_base, f'EEG_trial_{i}.npy')
                goni_file = os.path.join(self.data_base, f'goni_trial_{i}.npy')
                phase_file = os.path.join(self.data_base, f'phase_trial_{i}.npy')
                EEG = np.load(EEG_file).astype(np.float32)
                goni = np.load(goni_file).astype(np.float32)
                phase = np.load(phase_file).astype(np.float32)
                self.EEG.append(EEG)
                self.goni.append(goni)
                self.phase.append(phase)
        self.EEG = np.concatenate(self.EEG, axis=0)
        self.goni = np.concatenate(self.goni, axis=0)
        self.phase = np.concatenate(self.phase, axis=0)
        ch_names = ['FP1', 'FZ', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1',
                'PZ', 'P3', 'P7', 'O1', 'OZ', 'O2', 'P4', 'P8', 'CP6', 'CP2',
                'CZ', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'FP2', 'AF7', 'AF3',
                'AFZ', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1',
                'P5', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'P6', 'P2', 'CPZ', 'CP4',
                'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2']
        self.EEG_input_chans = self.get_chans(ch_names)
    
    def get_chans(self, ch_names):
        chans = []
        for ch_name in ch_names:
            chans.append(standard_1020.index(ch_name))
        return chans

    def __len__(self):
        return self.EEG.shape[0]

    def __getitem__(self, index):
        sample = {}
        EEG_X = self.EEG[index]
        Y = self.goni[index]
        phase = self.phase[index]
        Y = Y[-1, :]
        sample['EEG_X'] = EEG_X
        sample['Y'] = Y
        sample['phase'] = phase
        return sample

class NPYLoader_MoBI_ss(Dataset):
    def __init__(self, data_base, session2use, file_name='train_eeg.npy'):
        self.root = os.path.join(data_base, session2use)
        self.EEG = np.load(os.path.join(self.root, file_name))
        self.goni = np.load(os.path.join(self.root, file_name.replace('eeg', 'goni')))
        print(self.EEG.shape, self.goni.shape)
                        
        ch_names = ['FP1', 'FZ', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1',
                'PZ', 'P3', 'P7', 'O1', 'OZ', 'O2', 'P4', 'P8', 'CP6', 'CP2',
                'CZ', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'FP2', 'AF7', 'AF3',
                'AFZ', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1',
                'P5', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'P6', 'P2', 'CPZ', 'CP4',
                'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2']
        self.EEG_input_chans = self.get_chans(ch_names)
    
    def get_chans(self, ch_names):
        chans = []
        for ch_name in ch_names:
            chans.append(standard_1020.index(ch_name))
        return chans

    def __len__(self):
        return self.EEG.shape[0]

    def __getitem__(self, index):
        sample = {}
        EEG_X = self.EEG[index]
        Y = self.goni[index]
        Y = Y[-1, :]

        Y = torch.FloatTensor(Y)

        
        sample['EEG_X'] = EEG_X
        sample['Y'] = Y

        return sample

class PickleLoader(Dataset):
    def __init__(self, root, sessions2use, sub_dir='normal', sort=False, is_train=True):
        self.root = root
        self.sessions = os.listdir(root)
        self.train_files = []
        self.is_train = is_train
        
        for session in self.sessions:
            if session in sessions2use:
                if sub_dir is not None:
                    for file in os.listdir(os.path.join(root, session, sub_dir)):
                        
                        full_path = os.path.join(root, session, sub_dir, file)
                        
                        if os.path.getsize(full_path) > 0:
                            self.train_files.append(full_path)
                else:
                    for file in os.listdir(os.path.join(root, session)):
                        
                        full_path = os.path.join(root, session, file)
                        
                        if os.path.getsize(full_path) > 0:
                            self.train_files.append(full_path)

                if sort:
                    self.train_files = sorted(self.train_files, key=extract_session_and_number)
                    
                        
        ch_names = ['FP1', 'FZ', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1',
                'PZ', 'P3', 'P7', 'O1', 'OZ', 'O2', 'P4', 'P8', 'CP6', 'CP2',
                'CZ', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'FP2', 'AF7', 'AF3',
                'AFZ', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1',
                'P5', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'P6', 'P2', 'CPZ', 'CP4',
                'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2']
        self.EEG_input_chans = self.get_chans(ch_names)
    
    def get_chans(self, ch_names):
        chans = []
        for ch_name in ch_names:
            chans.append(standard_1020.index(ch_name))
        return chans

    def __len__(self):
        return len(self.train_files)

    def __getitem__(self, index):
        file_path = self.train_files[index]
        path = os.path.normpath(file_path) 

        parts = path.split(os.sep)  
        session = parts[-2].split('_')
        domain = int(session[0]) * 2 + int(session[1]) - 3
        domain = torch.tensor(domain, dtype=torch.int64) 
        
        try:
            with open(file_path, "rb") as f:
                sample = pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"⚠️ Warning: Skipping corrupted file {file_path}")
            return self.__getitem__((index + 1) % len(self.train_files)) 
        
        EEG_X = sample["EEG"]#(59, 400)

        Y = sample["goni"] #(400, 6)
        phase = sample["phase"] #(400)

        EEG_X = torch.FloatTensor(EEG_X)
        EEG_X = EEG_X - EEG_X.mean(dim=0, keepdim=True)   
        Y = torch.FloatTensor(Y)
        if not self.is_train:
            Y = Y[-1, :]

        
        sample.pop('EEG', None)

        sample['EEG_X'] = EEG_X
        sample['Y'] = Y
        sample['phase'] = phase
        sample['domain'] = domain

        return sample

class PickleLoader_MoBI(Dataset):
    def __init__(self, root, sessions2use, sub_dir=None, sort=False, is_train=True):
        self.root = root
        self.sessions = os.listdir(root)
        self.train_files = []
        self.is_train = is_train
        
        for session in self.sessions:
            if session in sessions2use:
                if sub_dir is not None:
                    for file in os.listdir(os.path.join(root, session, sub_dir)):
                        
                        full_path = os.path.join(root, session, sub_dir, file)
                        
                        if os.path.getsize(full_path) > 0:
                            self.train_files.append(full_path)
                else:
                    # print(os.path.join(root, session))
                    for file in os.listdir(os.path.join(root, session)):
                        
                        full_path = os.path.join(root, session, file)
                        
                        if os.path.getsize(full_path) > 0:
                            self.train_files.append(full_path)

                if sort:
                    self.train_files = sorted(self.train_files, key=extract_session_and_number)
            
                    
                        
        ch_names = ['FP1', 'FZ', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1',
                'PZ', 'P3', 'P7', 'O1', 'OZ', 'O2', 'P4', 'P8', 'CP6', 'CP2',
                'CZ', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'FP2', 'AF7', 'AF3',
                'AFZ', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1',
                'P5', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'P6', 'P2', 'CPZ', 'CP4',
                'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2']
        self.EEG_input_chans = self.get_chans(ch_names)
    
    def get_chans(self, ch_names):
        chans = []
        for ch_name in ch_names:
            chans.append(standard_1020.index(ch_name))
        return chans

    def __len__(self):
        return len(self.train_files)

    def __getitem__(self, index):
        file_path = self.train_files[index]
        # print(f"Loading file: {file_path}")  # debug
        path = os.path.normpath(file_path) 

        parts = path.split(os.sep)  
        session = parts[-2].split('_')
        domain = int(session[0]) * 3 + int(session[1])
        domain = torch.tensor(domain, dtype=torch.int64) 
        
        try:
            with open(file_path, "rb") as f:
                sample = pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"⚠️ Warning: Skipping corrupted file {file_path}")
            return self.__getitem__((index + 1) % len(self.train_files))
        
        EEG_X = sample["EEG"]#(59, 400)
        Y = sample["goni"]


        EEG_X = torch.FloatTensor(EEG_X)
        Y = torch.FloatTensor(Y)
        if not self.is_train:
            Y = Y[-1, :]

        sample.pop('EEG', None)

        sample['EEG_X'] = EEG_X
        sample['Y'] = Y
        sample['domain'] = domain

        return sample

class PickleLoader_FBM(Dataset):
    def __init__(self, root, sessions2use, sub_dir=None, sort=False, is_train=True):
        self.root = root
        self.sessions = os.listdir(root)
        self.train_files = []
        self.is_train = is_train
        
        for session in self.sessions:
            if session in sessions2use:
                if sub_dir is not None:
                    for file in os.listdir(os.path.join(root, session, sub_dir)):
                        
                        full_path = os.path.join(root, session, sub_dir, file)
                        
                        if os.path.getsize(full_path) > 0:
                            self.train_files.append(full_path)
                else:
                    # print(os.path.join(root, session))
                    for file in os.listdir(os.path.join(root, session)):
                        
                        full_path = os.path.join(root, session, file)
                        
                        if os.path.getsize(full_path) > 0:
                            self.train_files.append(full_path)

                if sort:
                    self.train_files = sorted(self.train_files, key=extract_session_and_number)
            
                    
                        
        ch_names = ['FP1', 'FZ', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1',
                'PZ', 'P3', 'P7', 'O1', 'OZ', 'O2', 'P4', 'P8', 'CP6', 'CP2',
                'CZ', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'FP2', 'AF7', 'AF3',
                'AFZ', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1',
                'P5', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'P6', 'P2', 'CPZ', 'CP4',
                'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2']
        self.EEG_input_chans = self.get_chans(ch_names)
    
    def get_chans(self, ch_names):
        chans = []
        for ch_name in ch_names:
            chans.append(standard_1020.index(ch_name))
        return chans

    def __len__(self):
        return len(self.train_files)

    def __getitem__(self, index):
        file_path = self.train_files[index]
        # print(f"Loading file: {file_path}")  # debug
        path = os.path.normpath(file_path)

        parts = path.split(os.sep)  
        session = parts[-2].split('_')
        domain = int(session[0]) - 2
        domain = torch.tensor(domain, dtype=torch.int64) 
        
        try:
            with open(file_path, "rb") as f:
                sample = pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"⚠️ Warning: Skipping corrupted file {file_path}")
            return self.__getitem__((index + 1) % len(self.train_files))
        
        EEG_X = sample["EEG"]#(59, 400)
        Y = sample["Y"]

        EEG_X = torch.FloatTensor(EEG_X)
        Y = torch.FloatTensor(Y)
        if not self.is_train:
            Y = Y[-1, :]

        sample.pop('EEG', None)

        sample['EEG_X'] = EEG_X
        sample['Y'] = Y
        sample['domain'] = domain

        return sample


class PickleLoader_p2(Dataset):
    def __init__(self, root, sessions2use, sub_dir='normal', sort=False, mode='all'):
        self.root = root
        self.sessions = os.listdir(root)
        self.train_files = []
        start_pkl = 200
        end_pkl = 400
        
        for session in self.sessions:
            if session in sessions2use:
                print(session)
                if sub_dir is not None:
                    if mode == 'all':
                        for file in os.listdir(os.path.join(root, session, sub_dir)):
                            
                            full_path = os.path.join(root, session, sub_dir, file)
                            
                            if os.path.getsize(full_path) > 0:
                                self.train_files.append(full_path)
                    elif mode == 'vis':
                        for i in range(start_pkl, end_pkl):
                            full_path = os.path.join(root, session, sub_dir, f'{i}.pkl')
                            
                            if os.path.getsize(full_path) > 0:
                                self.train_files.append(full_path)
                       

                else:
                    if mode == 'all':
                        for file in os.listdir(os.path.join(root, session)):
                            
                            full_path = os.path.join(root, session, file)
                            
                            if os.path.getsize(full_path) > 0:
                                self.train_files.append(full_path)

                    elif mode == 'vis':
                        for i in range(start_pkl, end_pkl):
                            full_path = os.path.join(root, session, f'{i}.pkl')
                            
                            if os.path.getsize(full_path) > 0:
                                self.train_files.append(full_path)
                if sort == 'number':
                    self.train_files = sorted(self.train_files, key=extract_number_then_session)
                elif sort:
                    self.train_files = sorted(self.train_files, key=extract_session_and_number)
                
                    
                    
                        
        ch_names = ['FP1', 'FZ', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1',
                'PZ', 'P3', 'P7', 'O1', 'OZ', 'O2', 'P4', 'P8', 'CP6', 'CP2',
                'CZ', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'FP2', 'AF7', 'AF3',
                'AFZ', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1',
                'P5', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'P6', 'P2', 'CPZ', 'CP4',
                'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2']
        self.EEG_input_chans = self.get_chans(ch_names)
    
    def get_chans(self, ch_names):
        chans = []
        for ch_name in ch_names:
            chans.append(standard_1020.index(ch_name))
        return chans

    def __len__(self):
        return len(self.train_files)

    def __getitem__(self, index):
        file_path = self.train_files[index]
        # print(file_path)
        path = os.path.normpath(file_path)

        parts = path.split(os.sep)  
        session = parts[-2].split('_')
        domain = int(session[0]) * 2 + int(session[1]) - 3
        domain = torch.tensor(domain, dtype=torch.int64) 
        
        try:
            with open(file_path, "rb") as f:
                sample = pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"⚠️ Warning: Skipping corrupted file {file_path}")
            return self.__getitem__((index + 1) % len(self.train_files))
        
        EEG_X = sample["EEG"]#(59, 400)
        Y = sample["goni"] #(6)

        EEG_X = torch.FloatTensor(EEG_X)
        Y = torch.FloatTensor(Y)
        Y = Y[-1, :]

        sample.pop('EEG', None)
        sample['EEG_X'] = EEG_X
        sample['Y'] = Y
        sample['domain'] = domain

        return sample
    
class PickleLoader_MoBI_p2(Dataset):
    def __init__(self, root, sessions2use, sub_dir='normal', sort=False):
        self.root = root
        self.sessions = os.listdir(root)
        self.train_files = []
        
        for session in self.sessions:
            if session in sessions2use:
                if sub_dir is not None:
                    for file in os.listdir(os.path.join(root, session, sub_dir)):
                        
                        full_path = os.path.join(root, session, sub_dir, file)
                        
                        if os.path.getsize(full_path) > 0:
                            self.train_files.append(full_path)
                else:

                    for file in os.listdir(os.path.join(root, session)):
                        
                        full_path = os.path.join(root, session, file)
                        
                        if os.path.getsize(full_path) > 0:
                            self.train_files.append(full_path)

                if sort:
                    self.train_files = sorted(self.train_files, key=extract_session_and_number)
                    
                        
        ch_names = ['FP1', 'FZ', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1',
                'PZ', 'P3', 'P7', 'O1', 'OZ', 'O2', 'P4', 'P8', 'CP6', 'CP2',
                'CZ', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'FP2', 'AF7', 'AF3',
                'AFZ', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1',
                'P5', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'P6', 'P2', 'CPZ', 'CP4',
                'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2']
        self.EEG_input_chans = self.get_chans(ch_names)
    
    def get_chans(self, ch_names):
        chans = []
        for ch_name in ch_names:
            chans.append(standard_1020.index(ch_name))
        return chans

    def __len__(self):
        return len(self.train_files)

    def __getitem__(self, index):
        file_path = self.train_files[index]
        path = os.path.normpath(file_path)
        parts = path.split(os.sep)  
        session = parts[-2].split('_')
        domain = int(session[0]) * 3 + int(session[1])
        domain = torch.tensor(domain, dtype=torch.int64) 
        
        try:
            with open(file_path, "rb") as f:
                sample = pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"⚠️ Warning: Skipping corrupted file {file_path}")
            return self.__getitem__((index + 1) % len(self.train_files))
        
        EEG_X = sample["EEG"]#(59, 400)
        Y = sample["goni"] #(6)

        EEG_X = torch.FloatTensor(EEG_X)
        Y = torch.FloatTensor(Y)
        Y = Y[-1, :]
        

        sample.pop('EEG', None)

        sample['EEG_X'] = EEG_X
        sample['Y'] = Y
        sample['domain'] = domain

        return sample

class PickleLoader_FBM_p2(Dataset):
    def __init__(self, root, sessions2use, sub_dir='normal', sort=False):
        self.root = root
        self.sessions = os.listdir(root)
        self.train_files = []
        
        for session in self.sessions:
            if session in sessions2use:
                if sub_dir is not None:
                    for file in os.listdir(os.path.join(root, session, sub_dir)):
                        
                        full_path = os.path.join(root, session, sub_dir, file)
                        
                        if os.path.getsize(full_path) > 0:
                            self.train_files.append(full_path)
                else:

                    for file in os.listdir(os.path.join(root, session)):
                        
                        full_path = os.path.join(root, session, file)
                        
                        if os.path.getsize(full_path) > 0:
                            self.train_files.append(full_path)

                if sort:
                    self.train_files = sorted(self.train_files, key=extract_session_and_number)
                    
                        
        ch_names = ['FP1', 'FZ', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1',
                'PZ', 'P3', 'P7', 'O1', 'OZ', 'O2', 'P4', 'P8', 'CP6', 'CP2',
                'CZ', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'FP2', 'AF7', 'AF3',
                'AFZ', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1',
                'P5', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'P6', 'P2', 'CPZ', 'CP4',
                'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2']
        self.EEG_input_chans = self.get_chans(ch_names)
    
    def get_chans(self, ch_names):
        chans = []
        for ch_name in ch_names:
            chans.append(standard_1020.index(ch_name))
        return chans

    def __len__(self):
        return len(self.train_files)

    def __getitem__(self, index):
        file_path = self.train_files[index]
        path = os.path.normpath(file_path)

        parts = path.split(os.sep)  
        session = parts[-2].split('_')
        domain = int(session[0]) - 2
        domain = torch.tensor(domain, dtype=torch.int64) 
        
        try:
            with open(file_path, "rb") as f:
                sample = pickle.load(f)
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"⚠️ Warning: Skipping corrupted file {file_path}")
            return self.__getitem__((index + 1) % len(self.train_files)) 
        
        EEG_X = sample["EEG"]#(59, 400)
        Y = sample["Y"] #(6)

        EEG_X = torch.FloatTensor(EEG_X)
        Y = torch.FloatTensor(Y)
        Y = Y[-1, :]
        
        sample.pop('EEG', None)


        sample['EEG_X'] = EEG_X
        sample['Y'] = Y
        sample['domain'] = domain

        return sample
               
if __name__ == '__main__':
    root = '/home/intern/data/'
    val_session = '7_1'
    test_session = '7_2'
    dataset = PickleLoader(root, val_session, test_session)
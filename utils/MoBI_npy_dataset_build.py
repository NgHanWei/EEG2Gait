import mne
import numpy as np
import os
import pickle
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.signal import firwin, filtfilt
from scipy.signal import butter, sosfiltfilt
import numpy as np

def bandpass_iir_filter(eeg, sfreq, l_freq=0.1, h_freq=48, order=4):
    """
    Apply zero-phase IIR bandpass filter (Butterworth) to multichannel EEG data.

    Parameters:
    - eeg: ndarray, shape (n_samples, n_channels)
    - sfreq: float, sampling frequency
    - l_freq: float, low cutoff frequency in Hz
    - h_freq: float, high cutoff frequency in Hz
    - order: int, filter order (default=4)

    Returns:
    - filtered_eeg: ndarray, same shape as eeg
    """
    sos = butter(order, [l_freq, h_freq], btype='bandpass', fs=sfreq, output='sos')
    filtered_eeg = sosfiltfilt(sos, eeg, axis=0)
    return filtered_eeg

def bandpass_fir_filter(eeg, sfreq, l_freq=0.1, h_freq=48, filter_order=1001):
    """
    Apply zero-phase FIR bandpass filter to multichannel EEG data.

    Parameters:
    - eeg: ndarray, shape (n_samples, n_channels)
    - sfreq: float, sampling frequency (e.g., 200 Hz)
    - l_freq: float, low cutoff frequency in Hz
    - h_freq: float, high cutoff frequency in Hz
    - filter_order: int, number of taps in the FIR filter

    Returns:
    - filtered_eeg: ndarray, same shape as eeg
    """
    nyquist = sfreq / 2.0
    taps = firwin(filter_order, [l_freq / nyquist, h_freq / nyquist], pass_zero=False)

    # Apply zero-phase filtering along the time axis (axis=0)
    filtered_eeg = np.zeros_like(eeg)
    for ch in range(eeg.shape[1]):
        print(f'Filtering channel {ch + 1}/{eeg.shape[1]}')
        filtered_eeg[:, ch] = filtfilt(taps, 1.0, eeg[:, ch])
    
    return filtered_eeg

def make_samples(EEG, goni, step, window_size=1200):
    # EEG: (59, T)
    # goni: (T, 6)
    T = EEG.shape[1]
    samples = []
    for i in range(0, T - window_size, step):
        sample = {}
        sample['EEG'] = EEG[:, i:i+window_size]
        # sample['goni'] = goni[i:i+window_size]
        sample['goni'] = goni[i:i+window_size]  # Downsample goni from 200 Hz to 10 Hz
        samples.append(sample)
    return samples


def process_mobi_eeg(eeg):
    """
    Process raw MoBI EEG signal (T, 64) into standardized (T, 59) format:
    1. Remove FT9, FT10, PO9, PO10, TP9, TP10.
    2. Add AFZ = mean(AF3, AF4).
    3. Reorder channels to match GPP_ch_names.
    
    Parameters:
        eeg (np.ndarray): Raw EEG signal, shape (T, 64)
    
    Returns:
        processed_eeg (np.ndarray): Processed EEG, shape (T, 59)
    """

    # 原始 MoBI 通道顺序
    mobi_order = [
        'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
        'T7', 'C3', 'CZ', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
        'P7', 'P3', 'PZ', 'P4', 'P8', 'PO9', 'O1', 'OZ', 'O2', 'PO10', 'AF7', 'AF3',
        'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8',
        'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPZ', 'CP4', 'TP8', 'P5',
        'P1', 'P2', 'P6', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8'
    ]

    # GPP 标准通道顺序
    GPP_ch_names = ['FP1', 'FZ', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1',
                    'PZ', 'P3', 'P7', 'O1', 'OZ', 'O2', 'P4', 'P8', 'CP6', 'CP2',
                    'CZ', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'FP2', 'AF7', 'AF3',
                    'AFZ', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1',
                    'P5', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'P6', 'P2', 'CPZ', 'CP4',
                    'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2']

    # 要删除的通道
    remove_channels = ['FT9', 'FT10', 'PO9', 'PO10', 'TP9', 'TP10']

    # 1. 删除通道
    channel_to_idx = {ch: idx for idx, ch in enumerate(mobi_order)}
    keep_channels = [ch for ch in mobi_order if ch not in remove_channels]
    eeg = eeg[:, [channel_to_idx[ch] for ch in keep_channels]]

    # 2. 插入 AFZ = mean(AF3, AF4)
    af3_idx = keep_channels.index('AF3')
    af4_idx = keep_channels.index('AF4')
    afz = (eeg[:, af3_idx] + eeg[:, af4_idx]) / 2
    eeg = np.concatenate([eeg, afz[:, np.newaxis]], axis=1)
    keep_channels.append('AFZ')

    # 3. 按 GPP_ch_names 重新排序
    channel_to_idx_updated = {ch: idx for idx, ch in enumerate(keep_channels)}
    eeg = eeg[:, [channel_to_idx_updated[ch] for ch in GPP_ch_names]]

    return eeg  # shape (T, 59)





def make_samples_ss(EEG, goni, step, window_size=200):
    # EEG: (59, T)
    # goni: (T, 6)
    T = EEG.shape[1]
    eeg_all_train = []
    goni_all_train = []
    eeg_all_val = []
    goni_all_val = []
    eeg_all_test = []
    goni_all_test = []
    train_end = int(T / 20 * 13.5)
    val_end = int(T / 20 * 15)
    test_end = T
    for i in range(0, train_end - window_size, step):
        eeg_tmp = EEG[:, i:i+window_size]
        goni_tmp = goni[i:i+window_size]  # Downsample goni from 200 Hz to 10 Hz
        eeg_all_train.append(eeg_tmp)
        goni_all_train.append(goni_tmp)
    for i in range(train_end, val_end - window_size, step):
        eeg_tmp = EEG[:, i:i+window_size]
        goni_tmp = goni[i:i+window_size]  # Downsample goni from 200 Hz to 10 Hz
        eeg_all_val.append(eeg_tmp) 
        goni_all_val.append(goni_tmp)
    for i in range(val_end, test_end - window_size, step):
        eeg_tmp = EEG[:, i:i+window_size]
        goni_tmp = goni[i:i+window_size]  # Downsample goni from 200 Hz to 10 Hz
        eeg_all_test.append(eeg_tmp) 
        goni_all_test.append(goni_tmp)
    

    eeg_all_train = np.stack(eeg_all_train)
    goni_all_train = np.stack(goni_all_train)
    eeg_all_val = np.stack(eeg_all_val)
    goni_all_val = np.stack(goni_all_val)
    eeg_all_test = np.stack(eeg_all_test)
    goni_all_test = np.stack(goni_all_test)
    print(eeg_all_train.shape, goni_all_train.shape)
    return eeg_all_train, goni_all_train, eeg_all_val, goni_all_val, eeg_all_test, goni_all_test
base = r'X:\Gait_code_transfer_fuxi\Data\MoBI'
FT_path = base + '\\MoBI_npy_ss\\'
print(FT_path)
if not os.path.exists(FT_path):
    os.makedirs(FT_path)
# ch_names = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2']
original_sfreq = 100
target_sfreq = 100
raw_dir = base + '\\MoBI_rawarray\\'
print(raw_dir)
step = 1
window_size = 200
for sbj_id in range(0, 8):
    for session in range(3):
        with open('log_MoBI.txt', 'a') as f:
            f.write(f'{sbj_id} {session} start\n')
        session = str(session)
        flag = 0
        sbj_name = str(sbj_id)
        sess = f'S{sbj_name}T{session}.mat'
        eeg_path = os.path.join(raw_dir, 'X', sess)
        goni_path = os.path.join(raw_dir, 'y', sess)
        eeg_file_content = loadmat(eeg_path)
        goni_file_content = loadmat(goni_path)
        
        sbj_FT_path = os.path.join(FT_path, sbj_name + '_' + session)
        if not os.path.exists(sbj_FT_path):
            os.makedirs(sbj_FT_path)
        for mode in ['normal']:
            eeg = eeg_file_content['X'] #(T, 64)
            eeg = process_mobi_eeg(eeg)  # (T, 59)
            goni = goni_file_content['y']
            upsample_factor = target_sfreq // original_sfreq  # 2

            num_samples = eeg.shape[0]
            new_num_samples = num_samples * upsample_factor

            # 上采样 EEG 和 goni
            if upsample_factor != 1: 
                eeg = resample(eeg, new_num_samples, axis=0)
                goni_upsampled = resample(goni, new_num_samples, axis=0)
            else:
                eeg = eeg
                goni_upsampled = goni

            # mean_vals = np.mean(goni_upsampled, axis=0)  # (6,)
            # std_vals = np.std(goni_upsampled, axis=0)    # (6,)

            # std_vals[std_vals == 0] = 1  

            goni = goni_upsampled
            print('start bandpass filter')
            print(eeg.shape, goni.shape)
            eeg = bandpass_iir_filter(eeg, sfreq=target_sfreq, l_freq=0.1, h_freq=48) #(120000, 64)

                        
            # Common Average Referencing (CAR)
            # eeg = eeg - np.mean(eeg, axis=1, keepdims=True)
            eeg = np.transpose(eeg)
        
        
            eeg_all_train, goni_all_train, eeg_all_val, goni_all_val, eeg_all_test, goni_all_test = make_samples_ss(eeg, goni, step, window_size)
            np.save(os.path.join(sbj_FT_path, 'train_eeg.npy'), eeg_all_train)
            np.save(os.path.join(sbj_FT_path, 'train_goni.npy'), goni_all_train)
            np.save(os.path.join(sbj_FT_path, 'val_eeg.npy'), eeg_all_val)
            np.save(os.path.join(sbj_FT_path, 'val_goni.npy'), goni_all_val)
            np.save(os.path.join(sbj_FT_path, 'test_eeg.npy'), eeg_all_test)
            np.save(os.path.join(sbj_FT_path, 'test_goni.npy'), goni_all_test)
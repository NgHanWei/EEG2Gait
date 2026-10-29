import mne
import numpy as np
import os
import pickle
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.signal import decimate

def visualize_non_repeating(data):
    x_vals = []
    y_vals = []
    
    prev = None
    for idx, val in enumerate(data):
        if val != prev:
            x_vals.append(idx)
            y_vals.append(val)
            prev = val

    plt.figure(figsize=(15, 5))
    plt.plot(x_vals, y_vals, drawstyle='steps-post')
    plt.title("Phase Sequence (Skipped Repeats)")
    plt.xlabel("Index")
    plt.ylabel("Phase Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def detect_initial_pattern(phases, idx, max_lookahead=600):
    """
    查找合法的相位模式起点，如果失败返回 None 和 -1。
    返回：
        - pattern: 匹配到的合法模式（如 [1,2,3,0]）
        - start_idx: 模式实际起点索引
    """
    valid_patterns = [
        [0, 1, 2, 3],
        [1, 2, 3, 0],
        [2, 3, 0, 1],
        [3, 0, 1, 2]
    ]

    for look_start in range(idx, idx + max_lookahead, 20):
        if look_start + 4 > len(phases):
            break

        seq = []
        last = phases[look_start]
        seq.append(last)

        for i in range(look_start + 1, len(phases)):
            if phases[i] != last:
                seq.append(phases[i])
                last = phases[i]
                if len(seq) == 4:
                    break

        if len(seq) == 4:
            for pat in valid_patterns:
                if seq == pat:
                    return pat, look_start

    return None, -1  # 未找到合法模式
def sanitize_phase_strict(phase, start_idx=400):
    phase = np.array(phase)
    cleaned = phase.copy()
    n = len(phase)

    # 所有可能合法模式（循环）
    valid_patterns = [
        [0, 1, 2, 3],
        [1, 2, 3, 0],
        [2, 3, 0, 1],
        [3, 0, 1, 2]
    ]

    

    i = start_idx
    cleaned[:start_idx] = -1
    current_pattern, valid_start = detect_initial_pattern(phase, i)

    if current_pattern is None:
        # 整段不合法，标记为 -1
        print("❌ 无法识别合法初始相位模式")
        cleaned[start_idx:] = -1
        return cleaned
    else:
        # 把从 start_idx 到 valid_start 之间的所有点设为 -1
        cleaned[start_idx:valid_start] = -1
        i = valid_start
    flag = cleaned[i]
    for i in range(valid_start, n):
        
        if cleaned[i] == flag:
            continue
        elif cleaned[i] == (flag+1) % 4:
            flag = cleaned[i]
        else:
            return np.concatenate([cleaned[:i], sanitize_phase_strict(cleaned[i:], start_idx=0)])

    return cleaned
def find_adjusted_peaks(data, threshold_ratio=0.4, other_peaks_num=None):
    """
    Find peaks in `data` that exceed a dynamic threshold based on 0.3*min + 0.7*max.

    Parameters:
    - data: 1D array
    - other_peaks_num: int or None. If given, limit output to top-N spaced peaks.

    Returns:
    - List of peak indices.
    """
    data = data.flatten()
    percentile_75_act = np.min(data) * (1-threshold_ratio) + np.max(data) * threshold_ratio

    # print(percentile_75_act)
    # 找出所有大于均值一定比例的峰值
    # window_size = 10
    # smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    peaks, _ = find_peaks(data.flatten(), distance=20)

    # 如果提供了其他峰值数量，并且当前峰值数量较多，则调整峰值数量
    if other_peaks_num is not None and len(peaks) > other_peaks_num:
        # 计算所有峰值之间的距离
        peak_distances = np.diff(peaks)
        # 找到距离最短的峰值对
        min_dist_idx = np.argmin(peak_distances)
        # 只保留更早出现的峰值
        peaks_to_remove = peaks[min_dist_idx + 1]
        peaks = np.delete(peaks, np.where(peaks == peaks_to_remove))
        # 如果删除一个峰值后仍然多于other_peaks_num，则继续此过程
        while len(peaks) > other_peaks_num:
            peak_distances = np.diff(peaks)
            min_dist_idx = np.argmin(peak_distances)
            peaks_to_remove = peaks[min_dist_idx + 1]
            peaks = np.delete(peaks, np.where(peaks == peaks_to_remove))
    tmp_peaks = []
    for peak in peaks:
        if data[peak] > percentile_75_act:
            tmp_peaks.append(peak)

    return tmp_peaks

def align_joint_extrema(goni_smooth):
    lh_peaks = find_adjusted_peaks(goni_smooth[:, 0], threshold_ratio=0.6)
    lk_peaks = find_adjusted_peaks(goni_smooth[:, 1], threshold_ratio=0.6)
    rh_valleys = find_adjusted_peaks(-goni_smooth[:, 3], threshold_ratio=0.6)
    rk_valleys = find_adjusted_peaks(-goni_smooth[:, 4], threshold_ratio=0.6)

    all_events = []
    all_events += [(idx, 'left_hip') for idx in lh_peaks]
    all_events += [(idx, 'left_knee') for idx in lk_peaks]
    all_events += [(idx, 'right_hip') for idx in rh_valleys]
    all_events += [(idx, 'right_knee') for idx in rk_valleys]
    all_events.sort(key=lambda x: x[0])

    pattern = ['left_hip', 'left_knee', 'right_hip', 'right_knee']
    pattern_idx = 0
    result = {key: [] for key in pattern}

    for time_idx, label in all_events:
        if label == pattern[pattern_idx]:
            result[label].append(time_idx)
            pattern_idx = (pattern_idx + 1) % len(pattern)

    return result, lh_peaks, lk_peaks, rh_valleys, rk_valleys


def visualize_joint_extrema(goni_smooth, result, lh_peaks, lk_peaks, rh_valleys, rk_valleys, sampling_rate=None):
    time = np.arange(goni_smooth.shape[0])
    if sampling_rate:
        time = time / sampling_rate

    plt.figure(figsize=(15, 8))

    # 原始候选点
    plt.plot(time, goni_smooth[:, 0], label='Left Hip', color='tab:blue')
    plt.plot(time[lh_peaks], goni_smooth[lh_peaks, 0], 'o', color='blue', label='LH Peaks')

    plt.plot(time, goni_smooth[:, 3], label='Right Hip', color='tab:orange')
    plt.plot(time[rh_valleys], goni_smooth[rh_valleys, 3], 'v', color='orange', label='RH Valleys')

    plt.plot(time, goni_smooth[:, 1], label='Left Knee', color='tab:green')
    plt.plot(time[lk_peaks], goni_smooth[lk_peaks, 1], 'o', color='green', label='LK Peaks')

    plt.plot(time, goni_smooth[:, 4], label='Right Knee', color='tab:red')
    plt.plot(time[rk_valleys], goni_smooth[rk_valleys, 4], 'v', color='red', label='RK Valleys')

    # 匹配后保留的极值点（大号点/星号）
    plt.plot(time[result['left_hip']], goni_smooth[result['left_hip'], 0], '*', color='darkblue', markersize=12, label='LH Final')
    plt.plot(time[result['left_knee']], goni_smooth[result['left_knee'], 1], '*', color='darkgreen', markersize=12, label='LK Final')
    plt.plot(time[result['right_hip']], goni_smooth[result['right_hip'], 3], '*', color='darkorange', markersize=12, label='RH Final')
    plt.plot(time[result['right_knee']], goni_smooth[result['right_knee'], 4], '*', color='darkred', markersize=12, label='RK Final')

    plt.xlabel('Time (samples)' if not sampling_rate else 'Time (s)')
    plt.ylabel('Angle (degrees)')
    plt.title('Joint Angles with Candidate and Ordered Peaks/Valleys')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def peak_detector(ind_act, dim=1):
    # 找到实际数据的峰值
    peaks_act = find_adjusted_peaks(ind_act[:, dim], threshold_ratio=0.9)
    # 基于实际数据的峰值进行调整
    peaks_act_adjusted = peaks_act#adjust_peaks_within_range(peaks_act, ind_act[:, dim],

    period = (peaks_act_adjusted[-1] - peaks_act_adjusted[0]) / (len(peaks_act_adjusted) - 1)

    # print('peaks_act_adjusted', peaks_act_adjusted)
    peaks_act_final = peaks_act_adjusted
    print('peaks_act_final', peaks_act_final)
    # period = len(ind_pred[:, dim]) / len(peaks_act_fi
    return peaks_act_final     

def visualize_gait_phases(signal, leftknee_peaks, rightknee_valleys, lefthip_peaks=None, righthip_valleys=None):
    plt.figure(figsize=(12, 6))
    for i in [0, 3]:  # Iterate over each channel
        plt.plot(signal[:, i], label=f'Channel {i+1}')
    plt.scatter(leftknee_peaks, signal[leftknee_peaks, 1], color='red', label='leftknee_peaks', zorder=5)  # Highlight peaks on Channel 1
    if rightknee_valleys is not None:
        plt.scatter(rightknee_valleys, signal[rightknee_valleys, 4], color='green', label='rightknee_valleys', zorder=5)
    if lefthip_peaks is not None:
        plt.scatter(lefthip_peaks, signal[lefthip_peaks, 0], color='blue', label='lefthip_peaks', zorder=5)
    if righthip_valleys is not None:
        plt.scatter(righthip_valleys, signal[righthip_valleys, 3], color='orange', label='righthip_valleys', zorder=5)
    plt.title('')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.show()

def determine_phase(current_idx, lefthip_peaks, leftknee_peaks, righthip_valleys, rightknee_valleys):
    """
    根据当前索引前最近的关键事件，决定当前步态阶段。

    Phase:
        1 = Left Hip Peak
        2 = Left Knee Peak
        3 = Right Hip Valley
        4 = Right Knee Valley
    """
    last_event_indices = []

    # 找出 <= current_idx 的最大值（即最近一次事件）
    def latest_event(event_list):
        return max([idx for idx in event_list if idx <= current_idx], default=-1)

    last_event_indices.append(('left_hip', latest_event(lefthip_peaks)))
    last_event_indices.append(('left_knee', latest_event(leftknee_peaks)))
    last_event_indices.append(('right_hip', latest_event(righthip_valleys)))
    last_event_indices.append(('right_knee', latest_event(rightknee_valleys)))

    # 找出其中 index 最大的那一个
    last_event_indices = [(name, idx) for name, idx in last_event_indices if idx != -1]
    if not last_event_indices:
        return -1  # 没有任何事件

    latest_name, _ = max(last_event_indices, key=lambda x: x[1])
    phase_map = {
        'left_hip': 1,
        'left_knee': 2,
        'right_hip': 3,
        'right_knee': 0
    }
    return phase_map[latest_name]
    
def detect_phase_transitions(phase):
    transitions = [-1] * len(phase)
    trans_pair = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for i in range(1, len(phase)):
        if (phase[i - 1], phase[i]) in trans_pair:
            transitions[i] = phase[i]
    return transitions


def make_samples(EEG, goni, step, window_size=1200, phase=None, transitions=None):
    # EEG: (59, T)
    # goni: (T, 6)
    T = EEG.shape[1]
    samples = []
    for i in range(0, T - window_size, step):
        sample = {}
        sample['EEG'] = EEG[:, i:i+window_size]
        # sample['goni'] = goni[i:i+window_size]
        sample['goni'] = goni[i:i+window_size]  # Downsample goni from 200 Hz to 10 Hz
        if phase is not None:
            sample['phase'] = phase[i+window_size]
        else:
            sample['phase'] = -1
        if transitions is not None:
            sample['transitions'] = transitions[i+window_size]
        else:
            sample['transitions'] = -1
        samples.append(sample)
    return samples

def make_stacked_samples(EEG, goni, step, window_size=1200, phase=None, transitions=None):
    # EEG: (59, T)
    # goni: (T, 6)
    T = EEG.shape[1]

    EEG_list = []
    goni_list = []
    phase_list = []
    transitions_list = []

    for i in range(0, T - window_size, step):
        EEG_window = EEG[:, i:i+window_size]
        goni_window = goni[i:i+window_size]

        if phase is not None:
            phase_value = phase[i+window_size]
        else:
            phase_value = -1

        if transitions is not None:
            transitions_value = transitions[i+window_size]
        else:
            transitions_value = -1

        EEG_list.append(EEG_window)
        goni_list.append(goni_window)
        phase_list.append(phase_value)
        transitions_list.append(transitions_value)

    EEG_array = np.stack(EEG_list)                # (N, 59, window_size)
    goni_array = np.stack(goni_list)              # (N, window_size, 6)
    phase_array = np.array(phase_list)            # (N,)
    transitions_array = np.array(transitions_list)  # (N,)

    return EEG_array, goni_array, phase_array, transitions_array

name2num = {'Aung': '1', 'Chengxuan': '2', 'David': '3', 'Dingyi': '4', 'Fuxi': '5',
             'Hanna': '6', 'Huizhi': '7', 'James': '8', 'Jenny': '9', 'LiYong': '10',
             'Meiqian': '11', 'Ruixuan': '12', 'Rui': '13', 'Shangen': '14', 'Shuailei': '15',
             'Sunhao': '16', 'Xueyi': '17', 'Yidan': '18', 'Yuhao': '19', 'Yunfeng': '20',
             'Yuren': '21', 'Zhangsu': '22', 'Zhiman': '23', 'Zhiwei': '24', 'Changhao': '25', 
             'Chengru': '26', 'Donald': '27', 'Dongping': '28', 'Hanwei': '29', 'Hongyu': '30', 
             'Kairui': '31', 'Lixun': '32', 'Meilun': '33', 'Noemie': '34', 'Rosary': '35', 
             'Shuqi': '36', 'Tang': '37', 'Wenjin': '38', 'Xiaohao': '39', 'Xiaojing': '40', 
             'Ximing': '41', 'Yiruo': '42', 'Youquan': '43', 'Yuan': '44', 'Yueying': '45', 
             'Yuting': '46', 'Zequn': '47', 'Zheren': '48', 'Zhisheng': '49', 'Zhuoru': '50'}
ch_names = ['FP1', 'FZ', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1',
                'PZ', 'P3', 'P7', 'O1', 'OZ', 'O2', 'P4', 'P8', 'CP6', 'CP2',
                'CZ', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'FP2', 'AF7', 'AF3',
                'AFZ', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1',
                'P5', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'P6', 'P2', 'CPZ', 'CP4',
                'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2']
base = r'X:\Gait_code_transfer_fuxi\Data\GPP'
FT_path = base + '\\trialwise_dataset\\'
print(FT_path)
goni_reverse_list = ['7_2', '11_2', '12_2', '20_1', '21_1']
if not os.path.exists(FT_path):
    os.makedirs(FT_path)
# ch_names = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 'F5', 'FT7', 'FC3', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'AF8', 'AF4', 'F2']
original_sfreq = 1000
target_sfreq = 100
raw_dir = base + '\\BlockMat\\'
print(raw_dir)
normal_blocks = ['1', '2', '3']
step = 1
window_size = 200
for sbj_id in range(1, 51):
    for session in ['1', '2']:
        # if sbj_id == 49 and session == '2':
        #     continue
        with open('log_GPP.txt', 'a') as f:
            f.write(f'{sbj_id} {session} start\n')
        flag = {}
        sbj_name = str(sbj_id)
        new_sbj_name = sbj_name
        sess = sbj_name + '_' + session
        eeg_path = os.path.join(raw_dir, sess)
        
        for mode in ['normal', 'left', 'right', 'left_2sand']:
            flag[mode] = 0
        
        for mat_file in os.listdir(eeg_path):
            if not mat_file.endswith('.mat'):
                continue
            if not mat_file.startswith('Partitioned'):
                continue
            if not mat_file.split('_')[3] == 'block':
                continue
            block = mat_file.split('Partitioned_session_walking_block_')[1][:-4]
            print(f'Processing {sbj_name} {session} {block}')
            # if block not in ['1', '2', '3']:
            #     continue
            if block in normal_blocks:
                mode = 'normal'
            else:
                continue
            sbj_FT_path = os.path.join(FT_path, mode, new_sbj_name + '_' + session)
            if not os.path.exists(sbj_FT_path):
                os.makedirs(sbj_FT_path)
            print(sbj_FT_path)
            print('Loading ' + os.path.join(eeg_path, mat_file))
            
            file_content = loadmat(os.path.join(eeg_path, mat_file))
            
            goni = file_content['goniometer_data_all'] #[9, 18, 13, 7]
            if goni.shape[1] not in [9, 18, 13, 7]:
                print('\nError in ' + mat_file + '\n\n\n\n\n')
                continue
            if (goni.shape[1] > 15):
                goni = goni[:, 6:12]
            elif (goni.shape[1] > 12):
                goni = goni[:, [0,2,4,6,8,10]]
            else:
                goni = goni[:, :6]
            if new_sbj_name + '_' + session in goni_reverse_list:
                goni = goni[:, [3, 4, 5, 0, 1, 2]]
            goni = decimate(goni, original_sfreq // target_sfreq, axis=0, ftype='iir')
            # Save the processed EEG data for later use
            # eeg_save_path = os.path.join(base, 'EEG_preprocessed', f'{new_sbj_name}_{session}_{block}_processed.npy')
            # EEG = np.load(eeg_save_path)
            
                
            

            mean_vals = np.mean(goni, axis=0)  # (6,)
            std_vals = np.std(goni, axis=0)    # (6,)

            std_vals[std_vals == 0] = 1  

            goni = (goni - mean_vals) / std_vals
            
            EEG = file_content['eeg_data_all']


            location_file = os.path.join(raw_dir, sess, 'locations.ced')
            with open(location_file, 'r') as f:
                lines = f.readlines()
                for i in range(1, len(lines)):
                    ch_name = lines[i].split('\t')[1]
                    if ch_name.upper() not in ch_names:
                        EEG = np.delete(EEG, i-1, axis=0)
            if EEG.shape[0] != len(ch_names):
                print('\n\n\n\n\nError in ' + mat_file + '\n\n\n\n\n')
                
            # 创建 MNE 的 RawArray 对象
            info = mne.create_info(ch_names=ch_names, sfreq=original_sfreq, ch_types="eeg")
            raw = mne.io.RawArray(EEG, info)

            raw = raw.filter(l_freq=0.1, h_freq=48, n_jobs=5)
            raw = raw.resample(target_sfreq, n_jobs=5)
            EEG = raw.get_data()


            if abs(EEG.shape[1] - goni.shape[0] ) > 2:
                print('\n\n\n\n\nError in ' + eeg_path + mat_file)
                print('EEG shape:', EEG.shape)
                print('Goni shape:', goni.shape)


            
            trial_times = file_content['current_trial_time']
            if block in ['1','left', 'right', 'left_2sand']:
                start_idx = 0
            else:
                start_idx = (int(block)-1) * 40
            for i in range(start_idx, start_idx + trial_times.shape[0]):
                print(f'{i} trial')
                start = int(trial_times[i - start_idx,0] * target_sfreq // original_sfreq)
                end = int(trial_times[i - start_idx,1] * target_sfreq // original_sfreq)
                current_goni = goni[start:end]# (T, 6)
                current_EEG = EEG[:, start:end]#(59, T)
                goni_smooth = savgol_filter(current_goni, window_length=25, polyorder=3, axis=0)
            
                result, lh, lk, rh, rk = align_joint_extrema(goni_smooth)
                # visualize_joint_extrema(goni_smooth, result, lh, lk, rh, rk, sampling_rate=100)  # 例如采样率100Hz
                T = goni.shape[0]
                phase = [determine_phase(t, lh, lk, rh, rk) for t in range(T)]
                # visualize_non_repeating(phase)
                phase = sanitize_phase_strict(phase, start_idx=window_size)
                # visualize_non_repeating(phase)
                transitions = detect_phase_transitions(phase)
                # visualize_non_repeating(transitions)

                EEG_array, goni_array, phase_array, transitions_array = make_stacked_samples(current_EEG, current_goni, step, window_size, phase, transitions=transitions)
                print(EEG_array.shape)
                np.save(os.path.join(sbj_FT_path, f'EEG_trial_{i}.npy'), EEG_array)
                np.save(os.path.join(sbj_FT_path, f'goni_trial_{i}.npy'), goni_array)
                np.save(os.path.join(sbj_FT_path, f'phase_trial_{i}.npy'), phase_array)
                np.save(os.path.join(sbj_FT_path, f'transitions_trial_{i}.npy'), transitions_array)


                # samples = make_samples(current_EEG, current_goni, step, window_size, phase, transitions=transitions)
                # print(len(samples))
                # for j, sample in enumerate(samples):
                #     file_name = f'{flag[mode]}.pkl'
                #     flag[mode] += 1
                #     # print(f'Writing {sbj_name} {session} {block} {mode} sample {j} to {sbj_FT_path}/{file_name}')
                #     # with open(os.path.join(sbj_FT_path, file_name), 'wb') as f:
                #     #     pickle.dump(sample, f)
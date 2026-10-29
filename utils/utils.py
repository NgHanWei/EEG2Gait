"""
    Comments:   This is the utility file for running the main program
    ToDo:       * Make it into a class for better usuability
    **********************************************************************************
"""
import numpy as np
import os
import json
import shutil
from logging import StreamHandler, INFO, DEBUG, Formatter, FileHandler, getLogger
import pdb
import datetime
import math
import numpy as np
import os
import yaml
from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
# from metrics import binary_metrics_fn, multiclass_metrics_fn
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from scipy.stats import pearsonr


from pkl_dataset import NPYLoader, NPYLoader_MoBI_ss, PickleLoader, PickleLoader_p2, PickleLoader_MoBI, PickleLoader_FBM, PickleLoader_MoBI_p2, PickleLoader_FBM_p2


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def prepare_pretrain_dataset(config_path):
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    datasets_list = []
    datasets = config['datasets']
    for dataset in datasets:
        contain_EEG = dataset['contain_EEG']
        contain_EOG = dataset['contain_EOG']
        contain_ECG = dataset['contain_ECG']
        contain_EMG = dataset['contain_EMG']
        path = dataset['path']
        files = os.listdir(os.path.join(path))
        datasets_list.append(PickleLoader(path, files, contain_EEG, contain_EOG, contain_ECG, contain_EMG, VQ_training=True))
    return datasets_list


def prepare_pretrain_VQ_dataset(root, contain_EEG=True, contain_EOG=False, contain_ECG=False, contain_EMG=False):
    files = os.listdir(os.path.join(root))
    train_files = files[:int(0.9 * len(files))]
    val_files = files[int(0.9 * len(files)):]

    print(len(train_files), len(val_files))

    train_dataset = PickleLoader(root, train_files, contain_EEG, contain_EOG, contain_ECG, contain_EMG, VQ_training=True)
    val_dataset = PickleLoader(root, val_files, contain_EEG, contain_EOG, contain_ECG, contain_EMG, VQ_training=True)

    return train_dataset, val_dataset



def prepare_GPP_dataset(data_base, train_sessions, val_sessions, test_sessions, test_data_base=None):
    dataset_train = PickleLoader(data_base, sessions2use=train_sessions, sub_dir=None, sort=False, is_train=True)
    
    if test_data_base is not None:
        dataset_val = PickleLoader(test_data_base, sessions2use=val_sessions, sub_dir=None, sort=True, is_train=False)
        dataset_test = PickleLoader(test_data_base, sessions2use=test_sessions, sub_dir=None, sort=True, is_train=False)
    else:
        dataset_val = PickleLoader(data_base, sessions2use=val_sessions, sub_dir=None, sort=True, is_train=False)
        dataset_test = PickleLoader(data_base, sessions2use=test_sessions, sub_dir=None, sort=True, is_train=False)
    
    return dataset_train, dataset_val, dataset_test

def prepare_ss_dataset(data_base, session2use, train_trials, val_trials, test_trials):
    dataset_train = NPYLoader(data_base, session2use, 0, train_trials)
    dataset_val = NPYLoader(data_base, session2use, train_trials, train_trials + val_trials)
    dataset_test = NPYLoader(data_base, session2use, train_trials + val_trials, train_trials + val_trials + test_trials)
    return dataset_train, dataset_val, dataset_test

def prepare_MoBI_dataset(data_base, train_sessions, val_sessions, test_sessions, test_data_base=None):
    dataset_train = PickleLoader_MoBI(data_base, sessions2use=train_sessions, sub_dir=None, sort=False, is_train=True)
    
    if test_data_base is not None:
        dataset_val = PickleLoader_MoBI(test_data_base, sessions2use=val_sessions, sub_dir=None, sort=True, is_train=False)
        dataset_test = PickleLoader_MoBI(test_data_base, sessions2use=test_sessions, sub_dir=None, sort=True, is_train=False)
    else:
        dataset_val = PickleLoader_MoBI(data_base, sessions2use=val_sessions, sub_dir=None, sort=True, is_train=False)
        dataset_test = PickleLoader_MoBI(data_base, sessions2use=test_sessions, sub_dir=None, sort=True, is_train=False)
    
    return dataset_train, dataset_val, dataset_test

def prepare_MoBI_ss_dataset(data_base, session2use):
    dataset_train = NPYLoader_MoBI_ss(data_base, session2use, file_name='train_eeg.npy')
    dataset_val = NPYLoader_MoBI_ss(data_base, session2use, file_name='val_eeg.npy')
    dataset_test = NPYLoader_MoBI_ss(data_base, session2use, file_name='test_eeg.npy')
    return dataset_train, dataset_val, dataset_test

def prepare_FBM_dataset(data_base, train_sessions, val_sessions, test_sessions, test_data_base=None):
    dataset_train = PickleLoader_FBM(data_base, sessions2use=train_sessions, sub_dir=None, sort=False, is_train=True)
    
    if test_data_base is not None:
        dataset_val = PickleLoader_FBM(test_data_base, sessions2use=val_sessions, sub_dir=None, sort=True, is_train=False)
        dataset_test = PickleLoader_FBM(test_data_base, sessions2use=test_sessions, sub_dir=None, sort=True, is_train=False)
    else:
        dataset_val = PickleLoader_FBM(data_base, sessions2use=val_sessions, sub_dir=None, sort=True, is_train=False)
        dataset_test = PickleLoader_FBM(data_base, sessions2use=test_sessions, sub_dir=None, sort=True, is_train=False)
    
    return dataset_train, dataset_val, dataset_test

def prepare_GPP_dataset_vis(data_base, train_sessions, val_sessions, test_sessions):
    dataset_train = PickleLoader_p2(data_base, sessions2use=train_sessions, sub_dir=None, sort=False, mode='vis')
    
    dataset_val = PickleLoader_p2(data_base, sessions2use=val_sessions, sub_dir=None, sort=True, mode='vis')
    dataset_test = PickleLoader_p2(data_base, sessions2use=test_sessions, sub_dir=None, sort=True, mode='vis')
    
    return dataset_train, dataset_val, dataset_test

def prepare_GPP_dataset_p2(data_base, train_sessions, val_sessions, test_sessions):
    dataset_train = PickleLoader_p2(data_base, sessions2use=train_sessions, sub_dir=None, sort=False)
    
    dataset_val = PickleLoader_p2(data_base, sessions2use=val_sessions, sub_dir=None, sort=True)
    dataset_test = PickleLoader_p2(data_base, sessions2use=test_sessions, sub_dir=None, sort=True)
    
    return dataset_train, dataset_val, dataset_test

def prepare_GPP_dataset_ft(data_base, test_sessions):
    
    dataset_test = PickleLoader_p2(data_base, sessions2use=test_sessions, sub_dir=None, sort='number')
    
    return dataset_test


def prepare_MoBI_dataset_p2(data_base, train_sessions, val_sessions, test_sessions):
    dataset_train = PickleLoader_MoBI_p2(data_base, sessions2use=train_sessions, sub_dir=None, sort=False)
    
    dataset_val = PickleLoader_MoBI_p2(data_base, sessions2use=val_sessions, sub_dir=None, sort=True)
    dataset_test = PickleLoader_MoBI_p2(data_base, sessions2use=test_sessions, sub_dir=None, sort=True)
    
    return dataset_train, dataset_val, dataset_test

def prepare_FBM_dataset_p2(data_base, train_sessions, val_sessions, test_sessions):
    dataset_train = PickleLoader_FBM_p2(data_base, sessions2use=train_sessions, sub_dir=None, sort=False)
    
    dataset_val = PickleLoader_FBM_p2(data_base, sessions2use=val_sessions, sub_dir=None, sort=True)
    dataset_test = PickleLoader_FBM_p2(data_base, sessions2use=test_sessions, sub_dir=None, sort=True)
    
    return dataset_train, dataset_val, dataset_test


def performance_metrics(Y, pre_Y, method):
    assert Y.shape == pre_Y.shape

    if method == 'r2':
        score = list(r2_score(Y, pre_Y, multioutput='raw_values'))
        
    elif method == 'rmse':
        score = list(root_mean_squared_error(Y, pre_Y, multioutput='raw_values'))
    elif method == 'pearsonr':
        score = [pearsonr(Y[:, idx],pre_Y[:,idx])[0] for idx in range(Y.shape[1])]
    elif method == 'mae':
        score = list(mean_absolute_error(Y, pre_Y, multioutput='raw_values'))
    else:
        print('Error! {} is undefined.'.format(method))
    score = sum(score) / len(score)
    return score


def get_metrics(output, target, metrics, is_binary):
    if 'r2' in metrics:
        results = {}
        for metric in metrics:
            results[metric] = performance_metrics(target, output, metric)
        return results
    if is_binary:
        if 'roc_auc' not in metrics or sum(target) * (len(target) - sum(target)) != 0:  # to prevent all 0 or all 1 and raise the AUROC error
            results = binary_metrics_fn(
                target,
                output,
                metrics=metrics
            )
        else:
            results = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
    else:
        results = multiclass_metrics_fn(
            target, output, metrics=metrics
        )
    return results

def get_metrics_single(output, target, metrics, is_binary):
    if 'r2' in metrics:
        results = {}
        mean_vals = np.mean(target, axis=0)  # (6,)
        std_vals = np.std(target, axis=0)    # (6,)

        std_vals[std_vals == 0] = 1  

        target = (target - mean_vals) / std_vals
            
        for metric in metrics:
            for i in range(output.shape[1]):
                
                results[metric + '_' + str(i)] = performance_metrics(target[:, [i]], output[:, [i]], metric)
                if metric == 'r2':
                    import matplotlib.pyplot as plt
                    plt.figure()
                    plt.plot(target[1000:2000, [i]], label='True', marker='o')
                    plt.plot(output[1000:2000, [i]], label='Predicted', marker='x')
                    plt.legend()
                    plt.title("True vs Predicted")
                    plt.savefig(f"r2_score_{i}.png")
            # results[metric] = performance_metrics(target, output, metric)
        return results
    if is_binary:
        if 'roc_auc' not in metrics or sum(target) * (len(target) - sum(target)) != 0:  # to prevent all 0 or all 1 and raise the AUROC error
            results = binary_metrics_fn(
                target,
                output,
                metrics=metrics
            )
        else:
            results = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
    else:
        results = multiclass_metrics_fn(
            target, output, metrics=metrics
        )
    return results

def get_KL_list(kl_file, test_sess, KL_thres=40):
    with open(kl_file, 'r') as file:
        kl_divergences_str_keys = json.load(file)
    min_delete = 15
    max_delete = 50
    tmp = {}
    for key in kl_divergences_str_keys:
        # print(key)
        if key.split('-')[0] == test_sess:
            tmp[key] = kl_divergences_str_keys[key]
    sorted_dict_desc = {k: v for k, v in sorted(tmp.items(), key=lambda item: item[1], reverse=True)}
    values_list = list(sorted_dict_desc.values())

    # 检查是否有足够的元素
    if len(values_list) >= min_delete:
        tenth_largest_value = values_list[min_delete - 1]  # 因为索引是从0开始的
    else:
        tenth_largest_value = None
    if len(values_list) > max_delete:
        twenty_largest_value = values_list[max_delete - 1]  # 因为索引是从0开始的
    else:
        twenty_largest_value = None
    # print(sorted_dict_desc)
    sorted_dict_desc = {key: value for key, value in sorted_dict_desc.items() if value > min(KL_thres, tenth_largest_value) or value > max(KL_thres, twenty_largest_value)}
    problemetic_sess = []
    for key in sorted_dict_desc:
        problemetic_sess.append(key.split('-')[1])
    problemetic_sess = ['Dingyi_1' if item == 'Dingyi_2' else 'Dingyi_2' if item == 'Dingyi_3' else item for item in
                     problemetic_sess]

    return problemetic_sess


def get_used_idx(s_names, problematic_sess, sample_per_epoch, val_sess, test_sess):
    bad_idx = []
    all_sess = []
    sep_list = val_sess + test_sess
    for name in s_names:
        if name + '_1' not in sep_list:
            all_sess.append(name + '_1')
        if name + '_2' not in sep_list:
            all_sess.append(name + '_2')
    for sess in problematic_sess:
        if sess in sep_list:
            continue
        sess_idx = all_sess.index(sess)
        start_idx = sess_idx * sample_per_epoch
        bad_idx.extend(range(start_idx, start_idx + sample_per_epoch))

    # 将 bad_idx 转换为集合
    bad_idx_set = set(bad_idx)

    # 使用集合来过滤 use_idx
    use_idx = [idx for idx in range(sample_per_epoch * len(all_sess)) if idx not in bad_idx_set]


    return use_idx


def load_data(mat_path, sess, start=0, end=20000):

    for file in os.listdir(mat_path):

        if file.endswith('.npy'):
            sbj = file.split('_')[0]
            session = file.split('_')[1]
            if sbj + '_' + session == sess:
                # 加载.npy文件
                print(file)
                file_path = os.path.join(mat_path, file)
                print("before load:", datetime.datetime.now().strftime("%H:%M:%S"))
                data = np.load(file_path)[start: end]
                print("after load:", datetime.datetime.now().strftime("%H:%M:%S"))
                lable_path = mat_path.replace('eeg', 'joint')
                label_file = file.replace('EEG', 'Joint')
                label = np.load(os.path.join(lable_path, label_file))[start: end]

                return data, label

def sample_load_data(mat_path ,sess, sample_per_sess):
    for file in os.listdir(mat_path):

        if file.endswith('.npy'):
            sbj = file.split('_')[0]
            session = file.split('_')[1]
            if sbj + '_' + session == sess:
                # 加载.npy文件
                print(file)
                file_path = os.path.join(mat_path, file)
                print("before load:", datetime.datetime.now().strftime("%H:%M:%S"))
                data = np.load(file_path)
                print("after load:", datetime.datetime.now().strftime("%H:%M:%S"))
                lable_path = mat_path.replace('eeg', 'joint')
                label_file = file.replace('EEG', 'Joint')
                label = np.load(os.path.join(lable_path, label_file))
                # 确保每个文件中样本数量足够
                num_samples = data.shape[0]
                if num_samples >= sample_per_sess:
                    # 选择一个随机的起始点
                    start_index = np.random.randint(0, num_samples - 40000)
                    # 从起始点开始选择连续的40000个样本
                    sampled_data = data[start_index:start_index + 40000]
                    sampled_label = label[start_index:start_index + 40000]
                return sampled_data, sampled_label

def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()
def dir_maker(path, clean=False):
    """Creating folder structures based on the path specified.

    """

    # First check if the directory exists
    if path.exists():
        print("Path already exists.")
        if clean:
            print("Cleaning")
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
    else:
        print("Creating new folders.")
        # Create paths including parent directories
        path.mkdir(parents=True, exist_ok=True)


def set_logger(SAVE_OUTPUT, LOG_FILE_NAME):
    """For better handling logging functionality.

    Obtained and modified from Best practices to log from CS230 Deep Learning, Stanford.
    https://cs230-stanford.github.io/logging-hyperparams.html

    Example:
    ```
    logging.info("Starting training...")
    ```

    Attributes:
        SAVE_OUTPUT: The directory of where you want to save the logs
        LOG_FILE_NAME: The name of the log file

    Returns:
        logger: logger with the settings you specified.
    """

    logger = getLogger()
    logger.setLevel(INFO)

    if not logger.handlers:
        # Define settings for logging
        log_format = Formatter(
            '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
        # for streaming, up to INFO level
        handler = StreamHandler()
        handler.setLevel(DEBUG)
        handler.setFormatter(log_format)
        logger.addHandler(handler)

        # for file, up to DEBUG level
        handler = FileHandler(SAVE_OUTPUT + '/' + LOG_FILE_NAME, 'w')
        handler.setLevel(DEBUG)
        handler.setFormatter(log_format)
        logger.setLevel(DEBUG)
        logger.addHandler(handler)

    return logger


def timer(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)

    return int(hours), int(minutes), seconds


def chunking(x, y, batch_size, future_step, num_chan, num_chan_kin):
    """
    A function to chunk the data into a batch size
    :param x: Input data
    :param y: Output target
    :param future_step: How much further you want to predict
    :return: Chunked matrices both for input and output
    """
    # Initialize the sequence and the next value
    seq, next_val = [], []
    # seq = np.empty(shape=(len(x)-batch_size-future_step, batch_size, future_step))
    # next_val = np.empty(shape=(len(y)+batch_size+future_step-1, num_chan_kin))
    # Based on the batch size and the future step size,
    # run a for loop to create chunks.
    # Here, it's BATCH_SIZE - 1 because we are trying to predict
    # one sample ahead. You could change this to your own way
    # e.g. want to predict 5 samples ahead, then - 5
    for i in range(0, len(x) - batch_size - future_step, future_step):
        seq.append(x[i: i + batch_size, :])
        next_val.append(y[i + batch_size + future_step - 1, :])

    # So now the data is [Samples, Batch size, One step prediction]
    seq = np.reshape(seq, [-1, batch_size, num_chan])
    next_val = np.reshape(next_val, [-1, num_chan_kin])

    X = np.array(seq)
    Y = np.array(next_val)

    return X, Y


def update_args(args, best_params):
    """Update some of the parameters after optuna optimization.

    """

    if args.decode_type in args.rnn_decoders:
        # Regardless of fix_do, init_std is optimized so need to be updated
        args.init_std = float(best_params['init_std'])
        # If layers and hidden units are not fixed, it's optimized so update
        if args.fix_do == 0:
            args.rnn_num_hidden = int(best_params['rnn_num_hidden'])
            args.rnn_num_stack_layers = int(
                best_params['rnn_num_stack_layers'])
    # Do the same for TCN
    elif args.decode_type in args.cnn_decoders:
        args.tcn_num_hidden = int(best_params['tcn_num_hidden'])
        args.tcn_num_layers = int(best_params['tcn_num_layers'])
        args.tcn_kernel_size = int(best_params['tcn_kernel_size'])

    return args

if __name__ == '__main__':
    # 示例调用
    sample_per_epoch = 8000  # 假设每个会话的样本数为8000
    s_names = ['Aung', 'Changhao', 'Chengru', 'Chengxuan', 'David', 'Dingyi',
                   'Donald', 'Dongping', 'Fuxi', 'Hanna', 'Hanwei', 'Hongyu',
                   'Huizhi', 'James', 'Jenny', 'Kairui', 'LiYong', 'Lixun',
                   'Meilun', 'Meiqian', 'Noemie', 'Rosary', 'Rui', 'Ruixuan',
                   'Shangen', 'Shuailei', 'Shuqi', 'Sunhao', 'Tang', 'Wenjin',
                   'Xiaohao', 'Xiaojing', 'Ximing', 'Xueyi', 'Yidan', 'Yiruo',
                   'Youquan', 'Yuan', 'Yueying', 'Yuhao', 'Yunfeng', 'Yuren',
                   'Yuting', 'Zequn', 'Zhangsu', 'Zheren', 'Zhiman', 'Zhisheng',
                   'Zhiwei', 'Zhuoru']
    # problemetic_sess = ['Changhao_1', 'Kairui_1', 'Tang_2', 'Zequn_1', 'Yuan_1', 'Yidan_1', 'Ruixuan_1', 'Meiqian_1', 'Huizhi_2']
    for s_name in s_names:
        # if s_name not in [ 'Huizhi', 'Meiqian', 'Ruixuan']:
        #     continue
        # if s_name < 'Yuting':
        #     continue
        test_sess = s_name + '_2'
        val_sess = s_name + '_1'
        problemetic_sess = get_KL_list('../SbjInd/kl_dict.json', test_sess)
        print(problemetic_sess)
    use_idx = get_used_idx(s_names, problemetic_sess, sample_per_epoch, ['Changhao_1'], ['Aung_2'])
    print(use_idx)
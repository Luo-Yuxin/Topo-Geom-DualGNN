import os
import torch
import json
import shutil
from datetime import datetime


class ExperimentRecorder:
    def __init__(self, log_root='training_log'):
        self.log_root = log_root
        os.makedirs(self.log_root, exist_ok=True)
    
    def convert_config_to_serializable(self, config):
        new_config = {}
        for k, v in config.items():
            if isinstance(v, dict):
                new_config[k] = self.convert_config_to_serializable(v)
            elif isinstance(v, torch.Tensor):
                new_config[k] = v.tolist()
            elif isinstance(v, torch.device):
                new_config[k] = str(v)
            elif isinstance(v, (list, tuple)):
                new_list = []
                for item in v:
                    if isinstance(item, torch.Tensor):
                        new_list.append(item.tolist())
                    else:
                        new_list.append(item)
                new_config[k] = new_list
            else:
                new_config[k] = v
        return new_config

    def save_experiment_v2(self, config, timestamp, model_params_info, epoch_stats, val_stats, test_stats, 
                           best_model_path, last_model_path, best_val_acc_record, other_model_path={}, other_stats=None):
        # 1. 文件夹创建
        test_acc_str = f"{test_stats['Test_sem_acc']*100:.2f}"
        # timestamp = datetime.now().strftime('%m-%d-%H-%M')
        folder_name = f"{timestamp}_{test_acc_str}"
        save_dir = os.path.join(self.log_root, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        print(f"\n[Recorder] Creating experiment log at: {save_dir}")

        # 2. Config
        serializable_config = self.convert_config_to_serializable(config)
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(serializable_config, f, indent=4, ensure_ascii=False)

        # 3. Result
        result_data = {
            'timestamp': timestamp,
            'model_parameters': {
                'total': model_params_info,
                'trainable': model_params_info
            },
            'final_metrics': {
                'train_last_epoch': epoch_stats,
                'val_last_epoch': val_stats,
                'test_final': test_stats
            },
            'others': other_stats if other_stats is not None else []
        }
        with open(os.path.join(save_dir, 'result.json'), 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)

        # 4. Copy Models
        # Best Model
        if os.path.exists(best_model_path):
            best_acc_str = f"{best_val_acc_record*100:.2f}"
            new_best_name = f"best_val_sem_acc-{best_acc_str}.pth"
            shutil.copy(best_model_path, os.path.join(save_dir, new_best_name))
            print(f"[Recorder] Saved best model: {new_best_name}")
        
        # Last Model
        if os.path.exists(last_model_path):
            end_acc_str = f"{val_stats['Val_sem_acc']*100:.2f}"
            new_last_name = f"end_val_sem_acc-{end_acc_str}.pth"
            shutil.copy(last_model_path, os.path.join(save_dir, new_last_name))
            print(f"[Recorder] Saved last model: {new_last_name}")

        for k, v in other_model_path.items():
            name = k
            last_model_path = v
            if os.path.exists(last_model_path):
                new_last_name = f"{name}.pth"
                shutil.copy(last_model_path, os.path.join(save_dir, new_last_name))
                print(f"[Recorder] Saved other model: {new_last_name}")
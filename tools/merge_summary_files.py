import os

import numpy as np
from tbparse import SummaryReader
import matplotlib.pyplot as plt

DRAW_TYPE = "val_acc"   #[train_loss, val_acc]

if __name__ == "__main__":
    # keys = ["unetr", "vt_unet_b", "swin_unetr", "optimized_unet", "hrstnet_stage_2", "hrstnet_stage_3", "hrstnet_stage_4"]
    keys = ["unetr", "vt_unet_b", "swin_unetr", "optimized_unet", "extending_nnunet", "hrstnet_stage_2", "hrstnet_stage_3", "hrstnet_stage_4"]
    log_dir = "/home/albert_wei/Desktop/Revise_Sensors/outputs_results/"
    extending_results = {'0': 0.054389752, '209': 0.9259452819, '39': 0.904918968677, '109': 0.9231658577919006,
                         '129': 0.9244276285171509, '159': 0.9237457513809204, '219': 0.926937997341156,
                         '249': 0.9263838529586792, '279': 0.9263903498649597, '49': 0.9137964844703674,
                         '59': 0.9200201630592346, '69': 0.9194732308387756, '79': 0.9232448935508728,
                         '89': 0.9192858338356018, '9': 0.8628441095352173, '99': 0.9209408164024353,
                         '119': 0.9234777092933655, '139': 0.9232177734375, '149': 0.923186182975769,
                         '169': 0.9251490235328674, '179': 0.9269275665283203, '189': 0.9267491698265076,
                         '19': 0.8971754312515259, '199': 0.9247132539749146, '229': 0.9261509776115417,
                         '239': 0.9263573884963989, '259': 0.9264988303184509, '269': 0.9264882802963257,
                         '289': 0.9263198375701904, '29': 0.8927357196807861, '299': 0.9263559579849243}
    extending_results_2 = {'0': 0.054389752, '21': 0.9259452819, '4': 0.904918968677, '11': 0.9231658577919006,
                         '13': 0.9244276285171509, '16': 0.9237457513809204, '22': 0.926937997341156,
                         '25': 0.9263838529586792, '28': 0.9263903498649597, '5': 0.9137964844703674,
                         '6': 0.9200201630592346, '7': 0.9194732308387756, '8': 0.9232448935508728,
                         '9': 0.9192858338356018, '1': 0.8628441095352173, '10': 0.9209408164024353,
                         '12': 0.9234777092933655, '14': 0.9232177734375, '15': 0.923186182975769,
                         '17': 0.9251490235328674, '18': 0.9269275665283203, '19': 0.9267491698265076,
                         '2': 0.8971754312515259, '20': 0.9247132539749146, '23': 0.9261509776115417,
                         '24': 0.9263573884963989, '26': 0.9264988303184509, '27': 0.9264882802963257,
                         '29': 0.9263198375701904, '3': 0.8927357196807861, '30': 0.9263559579849243}
    extending_results_list = []
    for k in range(1, 30):
        extending_results_list.append(extending_results_2[str(k)])
    TRAINING_ACC_COUNT = 300
    VAL_ACC_COUNT = 29
    train_loss_dict = {}
    val_acc_dict = {}
    for log_f in os.listdir(log_dir):
        reader = SummaryReader(os.path.join(log_dir, log_f))
        df = reader.scalars
        # keys = df.columns.values
        # step_col = df['step']
        # tag_col = df['tag']
        # value_col = df['value']
        train_loss = []
        for i in range(TRAINING_ACC_COUNT):
            row = df.loc[i]
            step = row["step"]
            tag = row["tag"]
            val = row["value"]
            # print(step, tag, val)
            if i != step or tag != "train_loss":
                raise ValueError("value is not correct")
            train_loss.append(val)
        train_loss_dict[log_f] = train_loss

        if log_f == "extending_nnunet":
            continue
        val_acc = []
        for i in range(VAL_ACC_COUNT):
            row = df.loc[TRAINING_ACC_COUNT+i]
            step = row["step"]
            tag = row["tag"]
            val = row["value"]
            # print(step, tag, val)
            if tag != "val_acc":
                raise ValueError("value is not correct")
            val_acc.append(val)
        val_acc_dict[log_f] = val_acc

        # print(log_f, step_col.shape, tag_col.shape, value_col.shape, np.count_nonzero(tag_col == "train_loss"),
        #       np.count_nonzero(tag_col == "val_acc"))
    # print(train_loss_dict.keys(), val_acc_dict.keys())
    # for k, v in train_loss_dict.items():
    #     print(k, len(v))
    #
    # for k, v in val_acc_dict.items():
    #     print(k, len(v))

    if DRAW_TYPE == "train_loss":
        idx = np.arange(0, len(train_loss_dict[list(train_loss_dict.keys())[0]]))
        fig, ax = plt.subplots(1)
        for k in keys:
            plt.plot(idx, train_loss_dict[k], label=k, marker='s', linewidth=1, ms=0)  # fmt='o',
        fig.set_dpi(600.0)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Value')
        plt.legend(loc='upper right')
        plt.show()
    elif DRAW_TYPE == "val_acc":
        if "extending_nnunet" in keys:
            val_acc_dict["extending_nnunet"] = extending_results_list
        idx = np.arange(0, len(val_acc_dict[list(val_acc_dict.keys())[0]])) * 10
        fig, ax = plt.subplots(1)
        for k in keys:
            plt.plot(idx, val_acc_dict[k], label=k, marker='s', linewidth=1, ms=0)  # fmt='o',
        fig.set_dpi(600.0)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Accuracy')
        plt.legend(loc='lower right')
        plt.show()




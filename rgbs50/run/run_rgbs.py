import time
import torch
import numpy as np
from tqdm import tqdm
import multiprocessing
from ..dataset import *
from ..analysis import calc_rgbps_seq_performace

from ..utils.color_print import greenprint
from ..utils.read import imread, txtread


class ExperimentRGBS(object):
    def __init__(self, dataset_root, project_root):
        self.project_root = project_root
        self.dataset = RGBS50(dataset_root)

        self.save_root = os.path.join(self.project_root, 'output', self.dataset.name)

    def run(self, tracker, divided_init=True):
        greenprint(f'Running {tracker.name} on {self.dataset.name}')
        time.sleep(0.01)
        for seq_num, sequence in enumerate(self.dataset):

            rgb_boxes, s_boxes = [], []

            seq_save_root = os.path.join(self.save_root, tracker.name, sequence.name)
            seq_rgb_save = os.path.join(seq_save_root, 'light.txt')
            seq_s_save = os.path.join(seq_save_root, 'sonar.txt')

            if os.path.exists(seq_rgb_save) and os.path.exists(seq_s_save):
                continue
            else:
                try:
                    os.makedirs(seq_save_root)
                except:
                    pass
            for item_num, data in tqdm(enumerate(sequence), total=len(sequence),
                                       desc=f'Running: [{seq_num + 1}/{len(self.dataset)}]'):
                # read img
                light_img = imread(data['light'])
                sonar_img = imread(data['sonar'])

                # gt
                rgb_gt, s_gt = data['rgb_gt'], data['s_gt']

                # Run
                if item_num == 0:
                    # init
                    tracker.init(light_img, sonar_img, rgb_gt, s_gt)
                    rgb_box, s_box = rgb_gt, s_gt
                else:
                    # track
                    rgb_box, s_box = tracker.track(light_img, sonar_img)

                rgb_boxes.append(rgb_box)
                s_boxes.append(s_box)

            np.savetxt(seq_rgb_save, rgb_boxes, fmt='%d', delimiter=',')
            np.savetxt(seq_s_save, s_boxes, fmt='%d', delimiter=',')
        greenprint(f'Finish running {tracker.name} on {self.dataset.name}')

    def eval(self, tracker_name, protocol=1):
        time.sleep(0.01)
        rgb_succ_score_all, rgb_prec_score_all, rgb_norm_prec_score_all = [], [], []
        s_succ_score_all, s_prec_score_all, s_norm_prec_score_all = [], [], []

        for seq_num, sequence in enumerate(self.dataset):
            seq_save_root = os.path.join(self.save_root, (tracker_name), sequence.name)

            rgb_results = txtread(os.path.join(seq_save_root, 'light.txt'))
            s_results = txtread(os.path.join(seq_save_root, 'sonar.txt'))

            # print(sequence.name, len(rgb_results), len(sequence.rgb_gt))

            # test RGB results
            rgb_succ_score, rgb_prec_score, rgb_norm_prec_score = calc_rgbps_seq_performace(rgb_results,
                                                                                            sequence.rgb_gt,
                                                                                            protocol)

            rgb_succ_score_all.append(rgb_succ_score)
            rgb_prec_score_all.append(rgb_prec_score)
            rgb_norm_prec_score_all.append(rgb_norm_prec_score)

            # test Sonar results
            s_succ_score, s_prec_score, s_norm_prec_score = calc_rgbps_seq_performace(s_results,
                                                                                      sequence.sonar_gt,
                                                                                      protocol)

            s_succ_score_all.append(s_succ_score)
            s_prec_score_all.append(s_prec_score)
            s_norm_prec_score_all.append(s_norm_prec_score)

        rgb_succ_score = torch.tensor(rgb_succ_score_all).mean().tolist() * 100
        rgb_prec_score = torch.tensor(rgb_prec_score_all).mean().tolist() * 100
        rgb_norm_prec_score = torch.tensor(rgb_norm_prec_score_all).mean().tolist() * 100

        s_succ_score = torch.tensor(s_succ_score_all).mean().tolist() * 100
        s_prec_score = torch.tensor(s_prec_score_all).mean().tolist() * 100
        s_norm_prec_score = torch.tensor(s_norm_prec_score_all).mean().tolist() * 100
        greenprint(f'Eval [{tracker_name}] performance of [{self.dataset.name}]: ')
        print(f'\t\t\t\t | AUC Score   | Precision Score\t| Norm Precision Score\t |')
        print(
            f'[Light]\t\t\t | \t\t{rgb_succ_score:.1f}   | \t\t{rgb_prec_score:.1f}\t\t| \t\t{rgb_norm_prec_score:.1f}\t\t\t | ')
        print(
            f'[Sonar]\t\t\t | \t\t{s_succ_score:.1f}   | \t\t{s_prec_score:.1f}\t\t| \t\t{s_norm_prec_score:.1f}\t\t\t | ')

    def run_sequence(self, sequence, tracker_params, tracker_factory):
        tracker = tracker_factory(tracker_params)
        rgb_boxes, s_boxes = [], []

        seq_save_root = os.path.join(self.save_root, tracker.name, sequence.name)
        seq_rgb_save = os.path.join(seq_save_root, 'light.txt')
        seq_s_save = os.path.join(seq_save_root, 'sonar.txt')

        if os.path.exists(seq_rgb_save) and os.path.exists(seq_s_save):
            finish_num = int(len(os.listdir(os.path.join(self.save_root, tracker.name))))
            greenprint(f'\t[{finish_num}/50]Finish running {tracker.name} on {sequence.name}')
            return 0
        else:
            try:
                os.makedirs(seq_save_root)
            except:
                pass

        finish_num = int(len(os.listdir(os.path.join(self.save_root, tracker.name))))
        greenprint(f'[{finish_num}/50]Running Tracker: {tracker.name} Sequence: {sequence.name}')

        for item_num, data in enumerate(sequence):
            # read img
            try:
                light_img = imread(data['light'])
                sonar_img = imread(data['sonar'])
            except:
                print(data['light'])
                print(data['sonar'])
                raise

            # gt
            rgb_gt, s_gt = data['rgb_gt'], data['s_gt']

            # Run
            if item_num == 0:
                # init
                tracker.init(light_img, sonar_img, rgb_gt, s_gt)
                rgb_box, s_box = rgb_gt, s_gt
            else:
                # track
                rgb_box, s_box = tracker.track(light_img, sonar_img)

            rgb_boxes.append(rgb_box)
            s_boxes.append(s_box)

        np.savetxt(seq_rgb_save, rgb_boxes, fmt='%d', delimiter=',')
        np.savetxt(seq_s_save, s_boxes, fmt='%d', delimiter=',')
        finish_num = int(len(os.listdir(os.path.join(self.save_root, tracker.name))))
        greenprint(f'\t[{finish_num}/50]Finish running {tracker.name} on {sequence.name}')

    def multi_run(self, tracker_params, tracker_factory, threads):
        param_list = [(seq, tracker_params, tracker_factory) for seq in (self.dataset)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(self.run_sequence, param_list)

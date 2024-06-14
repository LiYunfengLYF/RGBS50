import os
import torch
import numpy as np


def split_zero(z, x):
    non_zero_rows = np.any(z != 0, axis=1)
    non_zero_z = z[non_zero_rows]
    non_zero_x = x[non_zero_rows]

    zero_z = z[~non_zero_rows]
    zero_x = x[~non_zero_rows]

    return non_zero_z, non_zero_x, zero_z, zero_x


def select_zero(x):
    zero_rows = np.any(x != 0, axis=1)
    zero_x = x[~zero_rows]
    return zero_x


def calc_iou(pred_bb, anno_bb):
    tl = torch.max(pred_bb[:, :2], anno_bb[:, :2])
    br = torch.min(pred_bb[:, :2] + pred_bb[:, 2:] - 1.0, anno_bb[:, :2] + anno_bb[:, 2:] - 1.0)
    sz = (br - tl + 1.0).clamp(0)

    # Area
    intersection = sz.prod(dim=1)
    union = pred_bb[:, 2:].prod(dim=1) + anno_bb[:, 2:].prod(dim=1) - intersection

    return (intersection / union)


def calc_rgbps_iou(result, gt, protocol=1):
    gt_non_zero, result_non_zero1, gt_zero, result_zero1 = split_zero(gt, result)

    # handle zero items of gt
    result_zero_length = len(select_zero(result_zero1))
    gt_zero_length = len(gt_zero)
    if gt_zero_length == 0:
        ones_score1 = zeros_score1 = []
    elif result_zero_length <= gt_zero_length:
        ones_score1 = [1 for i in range(result_zero_length)]
        zeros_score1 = [0 for i in range(gt_zero_length - result_zero_length)]
    else:
        raise f'gt_zero_length is less than result_zero_length'

    # handle zero items of result
    gt_non_zero2, result_non_zero2, gt_zero2, result_zero2 = split_zero(result_non_zero1, gt_non_zero)
    zeros_score2 = [0 for i in range(len(result_zero2))]

    iou = calc_iou(torch.tensor(result_non_zero2), torch.tensor(gt_non_zero2)).tolist()

    if protocol == 1:
        iou = iou + ones_score1 + zeros_score1 + zeros_score2
    elif protocol == 2:
        iou = iou
    else:
        raise 'only support protocol 1 and 2'

    return iou


def calc_rgbps_prec(result, gt, normalized=False, protocol=1):
    gt_non_zero, result_non_zero1, gt_zero, result_zero1 = split_zero(gt, result)

    # handle zero items of gt
    result_zero_length = len(select_zero(result_zero1))
    gt_zero_length = len(gt_zero)

    if gt_zero_length == 0:
        ones_score1 = zeros_score1 = []
    elif result_zero_length <= gt_zero_length:
        zeros_score1 = [0 for i in range(result_zero_length)]
        if normalized:
            ones_score1 = [5 for i in range(gt_zero_length - result_zero_length)]
        else:
            ones_score1 = [100 for i in range(gt_zero_length - result_zero_length)]
    else:
        # print(result_zero_length,gt_zero_length)
        raise f'gt_zero_length is less than result_zero_length'

    # handle zero items of result
    gt_non_zero2, result_non_zero2, gt_zero2, result_zero2 = split_zero(result_non_zero1, gt_non_zero)
    oness_score2 = [100 for i in range(len(result_zero2))]

    center_errors = etrack.calc_precision(torch.tensor(result_non_zero2), torch.tensor(gt_non_zero2),
                                          normalized=normalized).tolist()
    if protocol == 1:
        center_errors = center_errors + zeros_score1 + ones_score1 + oness_score2
    elif protocol == 2:
        center_errors = center_errors
    else:
        raise 'only support protocol 1 and 2'
    return center_errors


def calc_rgbps_curve(ious, center_errors, norm_center_errors):
    ious = np.asarray(ious, float)[:, np.newaxis]
    center_errors = np.asarray(center_errors, float)[:, np.newaxis]
    norm_center_errors = np.asarray(norm_center_errors, float)[:, np.newaxis]

    thr_iou = np.linspace(0, 1, 21)[np.newaxis, :]
    thr_ce = np.arange(0, 51)[np.newaxis, :]
    thr_ce_norm = np.arange(0, 51)[np.newaxis, :] / 100

    bin_iou = np.greater(ious, thr_iou)
    bin_ce = np.less_equal(center_errors, thr_ce)
    bin_norm_ce = np.less_equal(norm_center_errors, thr_ce_norm)

    succ_curve = np.mean(bin_iou, axis=0)
    prec_curve = np.mean(bin_ce, axis=0)
    norm_prec_curve = np.mean(bin_norm_ce, axis=0)

    return succ_curve, prec_curve, norm_prec_curve


def calc_rgbps_seq_performace(results_boxes, gt_boxes, protocol=1):
    assert len(results_boxes) == len(gt_boxes)

    center_errors = calc_rgbps_prec(results_boxes, gt_boxes, protocol=protocol)

    norm_enter_errors = calc_rgbps_prec(results_boxes, gt_boxes, normalized=True, protocol=protocol)
    ious = calc_rgbps_iou(results_boxes, gt_boxes, protocol=protocol)

    succ_curve, prec_curve, norm_prec_curve = calc_rgbps_curve(ious, center_errors, norm_enter_errors)

    succ_score = np.mean(succ_curve)
    prec_score = prec_curve[20]
    norm_prec_score = norm_prec_curve[20]

    return succ_score, prec_score, norm_prec_score


def calc_rgbps_seq_performace_plot(results_boxes, gt_boxes, protocol=1):
    assert len(results_boxes) == len(gt_boxes)

    center_errors = calc_rgbps_prec(results_boxes, gt_boxes, protocol=protocol)

    norm_enter_errors = calc_rgbps_prec(results_boxes, gt_boxes, normalized=True, protocol=protocol)
    ious = calc_rgbps_iou(results_boxes, gt_boxes, protocol=protocol)

    succ_curve, prec_curve, norm_prec_curve = calc_rgbps_curve(ious, center_errors, norm_enter_errors)

    return succ_curve, prec_curve, norm_prec_curve

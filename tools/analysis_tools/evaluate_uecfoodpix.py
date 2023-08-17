import cv2
import mmcv
import numpy as np
from mmengine.logging import print_log
from mmengine.utils import is_list_of
from terminaltables import AsciiTable
import os


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    if isinstance(pred_label, str):
        pred_label = np.load(pred_label)

    if isinstance(label, str):
        label = mmcv.imread(label, flag='unchanged', backend='pillow')
    # modify if custom classes
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        # avoid using underflow conversion
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label] #预测正确的区域 (num_, )
    area_intersect, _ = np.histogram( #计算直方图的函数， 计算预测正确的区域每个类别的个数, (num_class+1, )
        intersect, bins=np.arange(num_classes + 1)) # 加1是因为左闭右开
    area_pred_label, _ = np.histogram(   #计算预测结果中每个类别的个数
        pred_label, bins=np.arange(num_classes + 1))
    area_label, _ = np.histogram(label, bins=np.arange(num_classes + 1))#计算标签中每个类别的个数
    area_union = area_pred_label + area_label - area_intersect # 计算每一个类别预测与真实mask并起来的面积

    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):

    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = np.zeros((num_classes, ), dtype=float)
    total_area_union = np.zeros((num_classes, ), dtype=float)
    total_area_pred_label = np.zeros((num_classes, ), dtype=float)
    total_area_label = np.zeros((num_classes, ), dtype=float)
    for i in range(num_imgs):
        #intersect_and_union： 计算当前图像,在每一个类别上的交与并
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(results[i], gt_seg_maps[i], num_classes,
                                ignore_index, label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    #total_area_intersect, 所有图像每个类别预测正确的面积

    return total_area_intersect, total_area_union, \
        total_area_pred_label, total_area_label

def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False):

    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))
    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(results, gt_seg_maps,
                                                     num_classes, ignore_index,
                                                     label_map,
                                                     reduce_zero_label)
    # total_area_   所有图像在每个类别的预测与标签的交与并、预测、标签的面积
    all_acc = total_area_intersect.sum() / total_area_label.sum()  # 所有类别的macc
    acc = total_area_intersect / total_area_label                  # 每个类别的acc
    ret_metrics = [all_acc, acc]                            
    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union  # 每个类别的交并比
            ret_metrics.append(iou)
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            ret_metrics.append(dice)
    if nan_to_num is not None:
        ret_metrics = [
            np.nan_to_num(metric, nan=nan_to_num) for metric in ret_metrics
        ]
    return ret_metrics


def evaluate(results, gt_mask, class_names, metric='mIoU', logger=None, efficient_test=False, ignore_index=255):
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        
        num_classes = len(class_names) # 103
        ret_metrics = eval_metrics(
            results,
            gt_mask,
            len(class_names),
            ignore_index, #255
            metric,
            label_map=None, # None
            reduce_zero_label=False) # False

        class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']] # [['Class', 'IoU', 'Acc']]
        ret_metrics_round = [  # 取两位小数
            np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
        ]
        for i in range(num_classes): # 每个类别的iOU和acc
            class_table_data.append([class_names[i]] +
                                    [m[i] for m in ret_metrics_round[2:]] +
                                    [ret_metrics_round[1][i]])
        summary_table_data = [['Scope'] +
                              ['m' + head
                               for head in class_table_data[0][1:]] + ['aAcc']] # [['Scope', 'mIoU', 'mAcc', 'aAcc']]
        ret_metrics_mean = [  #[aAcc, mAcc, miou]
            np.round(np.nanmean(ret_metric) * 100, 2)
            for ret_metric in ret_metrics
        ]
        summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                                  [ret_metrics_mean[1]] +
                                  [ret_metrics_mean[0]]) #[['Scope', 'mIoU', 'mAcc', 'aAcc'], ['global', 43.9, 56.99, 83.0]]
        print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        print_log('\n' + table.table, logger=logger)
        print_log('Summary:', logger)
        table = AsciiTable(summary_table_data)
        print_log('\n' + table.table, logger=logger)

        for i in range(1, len(summary_table_data[0])):  
            eval_results[summary_table_data[0]
                         [i]] = summary_table_data[1][i] / 100.0  # {'mIoU': 0.439, 'mAcc': 0.5699000000000001, 'aAcc': 0.83}
        if is_list_of(results, str): #flase
            for file_name in results:
                os.remove(file_name)
        return eval_results


if __name__ == '__main__':

    dir = '/home/lanxing/mmsegmentation/deeplabv3+_uecfoodpix_pred_mask/vis_data/vis_image'
    for n,name in enumerate(os.listdir(dir)):
        items = name.split('.')
        if len(items) == 2:
            break;
        src = dir + "/"+name
        dst = dir + "/"+ items[0].split('_')[1] + ".png"
        os.rename(src, dst)

    gt_masks_path = "/mnt/data/lanxing/FOOD_CORRECT_SEG_DATASETS/UECFoodPIXCOMPLETE/test/mask"
    pred_masks_path = "/home/lanxing/mmsegmentation/deeplabv3+_uecfoodpix_pred_mask/vis_data/vis_image"

    files = os.listdir(gt_masks_path)
    preds = os.listdir(pred_masks_path)

    assert len(files) == len(preds)

    for i in range(len(files)):
        pred_path = os.path.join(pred_masks_path, files[i])
        assert os.path.exists(pred_path)

    
    gt_mask = []
    result = []
    for file in files:
        gt_mask_path = os.path.join(gt_masks_path, file)
        mask = cv2.imread(gt_mask_path)
        mask = mask[:, :, -1]
        pred_mask_path = os.path.join(pred_masks_path, file)
        pred_mask = cv2.imread(pred_mask_path)
        pred_mask = pred_mask[:, :, -1]
        assert pred_mask.shape == mask.shape
        gt_mask.append(mask)
        result.append(pred_mask)
    # result = mmcv.load('output_best/result.pkl') # predict results
    # gt_mask = mmcv.load('output_best/gt_mask.pkl')
    #class_names = [str(i) for i in range(1, 103)]
    class_names = ['background', 'rice', 'eels on rice', 'pilaf', 'chicken-\'n\'-egg on rice', 'pork cutlet on rice', 'beef curry', 'sushi', 'chicken rice',
                'fried rice', 'tempura bowl', 'bibimbap', 'toast', 'croissant', 'roll bread', 'raisin bread', 'chip butty',
                'hamburger', 'pizza', 'sandwiches', 'udon noodle', 'tempura udon', 'soba noodle', 'ramen noodle', 'beef noodle',
                'tensin noodle', 'fried noodle', 'spaghetti', 'Japanese-style pancake', 'takoyaki', 'gratin', 'sauteed vegetables', 'croquette',
                'grilled eggplant', 'sauteed spinach', 'vegetable tempura', 'miso soup', 'potage', 'sausage', 'oden', 'omelet',
                'ganmodoki', 'jiaozi', 'stew', 'teriyaki grilled fish', 'fried fish', 'grilled salmon', 'salmon meuniere', 'sashimi',
                'grilled pacific saury', 'sukiyaki', 'sweet and sour pork', 'lightly roasted fish', 'steamed egg hotchpotch', 'tempura', 'fried chicken', 'sirloin cutlet',
                'nanbanzuke', 'boiled fish', 'seasoned beef with potatoes', 'hambarg steak', 'beef steak', 'dried fish', 'ginger pork saute', 'spicy chili-flavored tofu',
                'yakitori', 'cabbage roll', 'rolled omelet', 'egg sunny-side up', 'fermented soybeans', 'cold tofu', 'egg roll', 'chilled noodle',
                'stir-fried beef and peppers', 'simmered pork', 'boiled chicken and vegetables', 'sashimi bowl', 'sushi bowl', 'fish-shaped pancake with bean jam', 'shrimp with chill source', 'roast chicken',
                'steamed meat dumpling', 'omelet with fried rice', 'cutlet curry', 'spaghetti meat sauce', 'fried shrimp', 'potato salad', 'green salad', 'macaroni salad',
                'Japanese tofu and vegetable chowder', 'pork miso soup', 'chinese soup', 'beef bowl', 'kinpira-style sauteed burdock', 'rice ball', 'pizza toast', 'dipping noodles',
                'hot dog', 'french fries', 'mixed rice', 'goya chanpuru', 'others', 'beverage']

    assert len(gt_mask) == len(result)
    for idx, name in enumerate(gt_mask):
        assert gt_mask[idx].shape == result[idx].shape


    eval_result = evaluate(result, gt_mask, class_names=class_names)
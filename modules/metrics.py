from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.spice.spice import Spice
import torch

def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res

# 单标签
def calculate_mlc_metrics(binary_predictions, binary_targets, threshold=0.5):
    """计算准确率、精确率、召回率和 F1 分数

    参数:
        - logits (torch.Tensor): 模型输出的 logits 张量,shape 为 (batch_size, num_classes)
        - labels (torch.Tensor): 标签张量,shape 为 (batch_size, num_classes)
        - threshold (float): 阈值用于将 logits 转换为二值化预测,默认为 0.5

    返回:
        - accuracy (torch.Tensor): 准确率,shape 为 (num_classes,)
        - precision (torch.Tensor): 精确率,shape 为 (num_classes,)
        - recall (torch.Tensor): 召回率,shape 为 (num_classes,)
        - f1_score (torch.Tensor): F1 分数,shape 为 (num_classes,)

    """
    
    # return label_correct   
    
    # # Convert predictions and targets to binary predictions
    # binary_predictions = torch.where(torch.sigmoid(preds) > threshold, 1, 0)
    # binary_targets = torch.where(targets > threshold, 1, 0)    
    
    # Calculate metrics for each label
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    
    for i in range(binary_targets.shape[1]):
        tp = ((binary_predictions[:, i] == 1) & (binary_targets[:, i] == 1)).sum().item()
        tn = ((binary_predictions[:, i] == 0) & (binary_targets[:, i] == 0)).sum().item()
        fp = ((binary_predictions[:, i] == 1) & (binary_targets[:, i] == 0)).sum().item()
        fn = ((binary_predictions[:, i] == 0) & (binary_targets[:, i] == 1)).sum().item()

        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
    # Calculate average metrics across all labels
    mean_accuracy = torch.mean(torch.tensor(accuracy_list))
    mean_precision = torch.mean(torch.tensor(precision_list))
    mean_recall = torch.mean(torch.tensor(recall_list))
    mean_f1 = torch.mean(torch.tensor(f1_list))

    return accuracy_list, mean_accuracy, precision_list, mean_precision, recall_list, mean_recall, f1_list, mean_f1


# 总标签
def prf_multilabel(total_preds, total_targets, threshold=0.5, average='macro'):
    
    predictions  = (total_preds > threshold).float()
    
    # 计算真正例数（TP）
    true_positives = (predictions  * total_targets).sum(dim=0)
    # 计算假正例数（FP）
    false_positives = (predictions  * (1 - total_targets)).sum(dim=0)
    # 计算假反例数（FN）
    false_negatives = ((1 - predictions ) * total_targets).sum(dim=0)
    
    # 计算精确率（precision）
    precision = true_positives / (true_positives + false_positives + 1e-9)
    # 计算召回率（recall）
    recall = true_positives / (true_positives + false_negatives + 1e-9)
    # 计算F1分数（f1）
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    
    # 计算样本数（support）
    support = total_targets.sum(dim=0)
    
    # 计算平均值
    if average == 'macro':
        precision = precision.mean()
        recall = recall.mean()
        f1 = f1.mean()
    elif average == 'weighted':
        weights = support / support.sum()
        precision = (precision * weights).sum()
        recall = (recall * weights).sum()
        f1 = (f1 * weights).sum()
    else:
        raise ValueError(f"Unsupported average '{average}'")
    
    # 返回精确率、召回率、F1分数和样本数
    return precision, recall, f1
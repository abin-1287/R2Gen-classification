import torch
import torch.nn as nn
import numpy as np


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


def compute_loss(output, reports_ids, reports_masks, image_feature, label):
    criterion = LanguageModelCriterion()
    cap_loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    samples_per_class = np.array([675, 198, 564, 614, 274, 591, 273, 204, 189])
    weights_inverse = 1 / samples_per_class
    weights_normalized = weights_inverse / weights_inverse.sum()
    class_weights = torch.tensor(
        weights_normalized, dtype=torch.float).to(label.device)
    class_criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    class_loss = class_criterion(image_feature, label)
    return cap_loss, class_loss, cap_loss+class_loss
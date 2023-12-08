import torch
import torch.nn as nn
import torch.nn.functional as F
#CrossEntropy Loss
class BootstrappedCE(nn.Module):
    def __init__(self, min_K, loss_th, ignore_index):
        super().__init__()
        self.K = min_K
        self.threshold = loss_th
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="none"
        )

    def forward(self, logits, labels):
        pixel_losses = self.criterion(logits, labels).contiguous().view(-1)

        mask=(pixel_losses>self.threshold)
        if torch.sum(mask).item()>self.K:
            pixel_losses=pixel_losses[mask]
        else:
            pixel_losses, _ = torch.topk(pixel_losses, self.K)
        return pixel_losses.mean()


class OhemCrossEntropy2d(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCrossEntropy2d, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


if __name__=='__main__':
    pass
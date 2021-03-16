from pytorch_lightning.metrics import Metric
import editdistance
import torch

class PER(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("edit_distance", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("target_length", default=torch.tensor(0), dist_reduce_fx="sum")
    def update(self, preds, targets):
        for i in range(len(preds)):
            ref_list, pred_list = list(filter(lambda a: a != 0, targets.tolist())), preds
            ref_list, pred_list = list(map(str, ref_list)), list(map(str, pred_list))
            self.edit_distance += editdistance.distance(ref_list, pred_list)
            self.target_length += len(ref_list)
    def compute(self):
        return self.edit_distance.float() / self.target_length if self.target_length > 0 else 0
from pytorch_lightning.metrics import Metrics
import editdistance

class PER(Metrics):
    def __init__(self):
        self.add_state("edit_distance", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("target_length", default=torch.tensor(0), dist_reduce_fx="sum")
    def update(self, preds, targets):
        for i in range(preds.shape[0]):
            ref_list, pred_list = targets.tolist().remove(0), preds.tolist().remove(0)
            self.edit_distance += editdistance(ref_list, pred_list)
            self.target_length += len(ref_list)
    def compute(self):
        return self.edit_distance.float() / self.target_length if self.target_length > 0 else 0
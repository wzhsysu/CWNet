from torch import nn


class MasterCriterion(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.key_pairs = cfg.criterion.key_pairs

        self.mod_dict = {}
        self.mod_dict['MixedIllumL1'] = MixedIllumL1(cfg)

    def forward(self, ret_dict, item_dict, ittr_dim_exist=False):
        loss_dict = {}
        total_loss = 0

        for loss_key in self.key_pairs:
            mod_key = self.cfg.criterion[loss_key].mod
            alpha = self.cfg.criterion[loss_key].alpha

            loss = self.mod_dict[mod_key](ret_dict, item_dict, ittr_dim_exist, loss_dict)
            loss_dict[loss_key] = loss
            total_loss += (alpha * loss)

        loss_dict["total_loss"] = total_loss

        return loss_dict


class MixedIllumL1(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.L1 = nn.L1Loss()

    def forward(self, pred_dict, gt_dict, ittr_dim_exist=False, loss_dict=None):
        pred = pred_dict["mixed_illum"]
        gt = gt_dict["gt_illum"]
        gt = gt.to(pred.device, non_blocking=True)
        if ittr_dim_exist:
            pred = pred[:, -1]

        loss = self.L1(pred, gt)

        return loss

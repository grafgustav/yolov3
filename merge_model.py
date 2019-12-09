import torch
import torch.nn as nn
import torch.nn.functional as F


class MergeModel(nn.Module):
    def __init__(self):
        '''
            Principles:
            Fixed Input with Zeropadding
            Input: [y, 12] = [xyxy, obj_conf, cls_1_conf, ..., cls_7_conf]
            y with zeropadding
            Output: [x, 7] = [xyxy, obj_conf, cls_conf, cls]

            # first flatten the whole thing
            # then shoot through some hidden linear layers
            # finally reshape output
        '''
        super(MergeModel, self).__init__()

        # input is actually expected to be (2, bs, 10647, 12)
        input_size = 255528  # this seems like A LOT of parameters
        output_size = 127764  # then reformat
        self.fc1 = nn.Linear(input_size, 127764)
        self.fc2 = nn.Linear(127764, 63882)
        self.fc3 = nn.Linear(63882, output_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # input are the predictions of Darknet
        t = torch.cat((x, y), dim=1)
        t.flatten()
        t = x.view(-1, self.num_flat_features(t))
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = F.relu(self.fc3(t))
        t.reshape((10647, 12))
        return t

    def trim_or_pad(self, x, goal_shape):
        # either trim or pad input (preferable we always pad though)
        np, pred_width = x.shape  # number of preds,
        max_preds, _ = goal_shape
        if np > max_preds:
            print('Shape out of bounds for this net')
            return -1
        elif np == max_preds:
            return x
        else:
            fill_preds = torch.zeros((max_preds - np, pred_width))
            x = torch.cat((x, fill_preds))
        return x




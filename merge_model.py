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

        # input is actually expected to be (bs, 240)
        input_size = 240
        output_size = 240  # then reformat
        self.fc1 = nn.Linear(input_size, 320)
        self.fc2 = nn.Linear(320, 160)
        self.fc3 = nn.Linear(160, output_size)

    def forward(self, x: torch.Tensor):
        # input are the predictions of Darknet
        t = x
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = F.relu(self.fc3(t))
        t = t.reshape((-1, 20, 12))
        return t

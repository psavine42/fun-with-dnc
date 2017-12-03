


import torch
from torch.autograd import Variable
import logger as sl
from utils import flat, repackage


def action_loss(logits, action, criterion, log=None):
    """
        Sum of losses of one hot vectors encoding an action
        :param logits: network output vector of [action, [[type_i, ent_i], for i in ents]]
        :param action: target vector size [7]
        :param criterion: loss function
        :return:
        """
    losses = []
    for idx, action_part in enumerate(flat(action)):
        tgt = Variable(torch.LongTensor([action_part]))
        losses.append(criterion(logits[idx], tgt))
    loss = torch.stack(losses, 0).sum()
    if log is not None:
        sl.log_loss(losses, loss)
    return loss


def logical_loss(logits, action, criterion):
    """
        some hand tunining of penalties for illegal actions...
            trying to force learning of types.

        action type => type_e...
        :param logits: network output vector of one_hot distributions
            [action, [type_i, ent_i], for i in ents]
        :param action: target vector size [7]
        :param criterion: loss function
        :return:
        """
    pass


def naive_loss(logits, targets, criterion, log=None):
    """
        Calculate best choice from among targets, and return loss

        :param logits:
        :param targets:
        :param criterion:
        :return: loss
        """
    loss_idx, _ = min(enumerate([action_loss(repackage(logits), a, criterion) for a in targets]))
    final_action = targets[loss_idx]
    return final_action, action_loss(logits, final_action, criterion, log=log)
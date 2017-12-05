import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pickle, os, glob

eps = 10e-6

def repackage(xs):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(xs) == Variable:
        return Variable(xs.data)
    else:
        return tuple(repackage(v) for v in xs)

def depackage(xs):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(xs) == Variable:
        return xs.data
    elif type(xs) == torch.Tensor:
        return xs
    else:
        return tuple(depackage(v) for v in xs)

def interface_part(num_reads, W):
                    #read_keys
    partition = [num_reads* W, num_reads, W, 1, W, W, num_reads, 1, 1, num_reads * 3]
    ds = []
    cntr = 0
    for idx in partition:
        tn = [cntr, cntr + idx]
        ds.append(tn)
        cntr += idx
    return ds


def running_avg(cum_correct, cum_total_move, last=-100):
    return sum(cum_correct[last:]) / sum(cum_total_move[last:])


def flat(container):
    acc = []
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flat(i):
                acc.append(j)
        else:
            acc.append(i)
    return acc


def show(tensors, m=""):
    print("--")
    if type(tensors) == torch.Tensor:
        print(m, tensors.size())
    else:
        print(m)
        [print(t.size()) for t in tensors]


def clean_runs(dirs, lim=10):
    def convert_bytes(num):
        for x in ['bytes', 'KB', 'MB']:
            if num < 1024.0:
                return "%3.1f %s" % (num, x)
            num /= 1024.0
    for d in glob.glob(dirs + '*'):
        if convert_bytes(os.stat(d).st_size) < lim:
            os.remove(d)


def dnc_checksum(state):
    return [state[i].data.sum() for i in range(6)]

def save(model, optimizer, lstm_state, time_stmp, args, itr):
    # out, mem, r_wghts, w_wghts, links, l_wghts, usage, (ho, hc) = state
    name =  str(time_stmp) + args.save + str(itr) + '.pkl'
    torch.save(model.state_dict(), args.prefix + 'models/dnc_model_' + name)
    torch.save(model, args.prefix + 'models/dnc_model_full_' + name)
    torch.save(lstm_state, args.prefix + 'models/lstm_state_' + name)
    torch.save(optimizer.state_dict(), args.prefix + 'models/optimizer_' + name)
    torch.save(optimizer, args.prefix + 'models/optimizer_full_' + name)
    print("Saving ... file {}".format( str(time_stmp) + args.save + str(itr) ))


def get_prediction(expanded_logits, idxs='all'):
    max_idxs = []
    if idxs == 'all':
        idxs = range(len(expanded_logits))
    for idx in idxs:
        _, pidx = expanded_logits[idx].data.topk(1)
        max_idxs.append(pidx.squeeze()[0])
    return tuple(max_idxs)

def to_human_readable(input, mask):
    # phase = gen.PHASES[mask]
    # expr = data.ix_to_expr(inputs)
    pass

def correct(pred, best_actions):
    #action_own = [pred[0], (pred[1], pred[2]), (pred[3], pred[4]), (pred[5], pred[6])]
    #correct = action_own in best_actions
    pass

def human_readable_res(Data, all_actions, best_actions, chosen_action, pred, Guided, loss_data):
    base_prob = 1 / len(all_actions)

    action_own = [pred[0], (pred[1], pred[2]), (pred[3], pred[4]), (pred[5], pred[6])]
    correct = action_own in best_actions
    best = 0
    action_der = None
    for action in [flat(a) for a in best_actions]:
        scores = [1 if pred[i] == action[i] else 0 for i in range(len(pred))]
        if sum(scores) >= best:
            best, action_der = sum(scores), scores

    # print(Data.one_hot_size)
    best_move_exprs = [Data.vec_to_expr(t) for t in best_actions]
    all_move_exprs = [Data.vec_to_expr(t) for t in all_actions]
    chos_move, crest = Data.vec_to_expr(action_own)
    # print("all     {}".format(', '.join(["{} {}".format(m[0], m[1]) for m in all_move_exprs])))
    print("best    {}".format(', '.join(["{} {}".format(m[0], m[1]) for m in best_move_exprs])))
    print("chosen: {} {}, guided {},  prob {:0.2f}, T? {}---loss {:0.4f}".format(
        chos_move, crest, Guided, base_prob, correct, loss_data
    ))
    return action_der


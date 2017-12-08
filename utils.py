import torch
import torch.cuda
from torch.autograd import Variable
import os, glob
from arg import args 

eps = 10e-6
MODEL_NAME = 'dnc_model.pkl'
OPTIM_NAME = 'optimizer.pkl'


def _variable(xs, **kwargs):
    if args.cuda is True:
        return Variable(xs, **kwargs).cuda()
    else:
        return Variable(xs, **kwargs)


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
    elif type(xs) == torch.cuda.FloatTensor:
        return xs
    else:
        return tuple(depackage(v) for v in xs)


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


def get_chkpt(name, idx=0):
    search = './models/*{}/checkpts/*'.format(name)
    chkpt = sorted(glob.glob(search))[-1] + '/'
    return chkpt + MODEL_NAME, chkpt + OPTIM_NAME


def dnc_checksum(state):
    return [state[i].data.sum() for i in range(6)]


def save(model, optimizer, lstm_state, args, global_epoch):
    chkpt = '{}{}/{}/'.format(args.base_dir, 'checkpts', global_epoch)
    os.mkdir(chkpt)

    torch.save(model.state_dict(), chkpt + MODEL_NAME)
    torch.save(model, chkpt + 'dnc_model_full.pkl')
    torch.save(lstm_state, chkpt + 'lstm_state.pkl')
    torch.save(optimizer.state_dict(), chkpt + OPTIM_NAME)
    torch.save(optimizer, chkpt + 'optimizer_full.pkl')
    print("Saving ... file...{}, chkpt_num:{}".format(args.base_dir, global_epoch))


def get_prediction(expanded_logits, idxs='all'):
    max_idxs = []
    if idxs == 'all':
        idxs = range(len(expanded_logits))
    for idx in idxs:
        _, pidx = expanded_logits[idx].data.topk(1)
        max_idxs.append(pidx.squeeze()[0])
    return tuple(max_idxs)


def closest_action(pred, actions):
    best, chosen_action = 0, None
    for action in [flat(a) for a in actions]:
        scores = [1 if pred[i] == action[i] else 0 for i in range(len(pred))]
        if sum(scores) >= best:
            best, chosen_action = sum(scores), scores
    return chosen_action


def human_readable_res(Data, all_actions, best_actions, pred, guided, loss_data):

    base_prob = 1 / len(all_actions)
    action_own = [pred[0], (pred[1], pred[2]), (pred[3], pred[4]), (pred[5], pred[6])]
    correct = action_own in best_actions

    action_der = closest_action(pred, best_actions)
    best_move_exprs = [Data.vec_to_expr(t) for t in best_actions]
    # all_move_exprs = [Data.vec_to_expr(t) for t in all_actions]
    chos_move, crest = Data.vec_to_expr(action_own)

    # print("all     {}".format(', '.join(["{} {}".format(m[0], m[1]) for m in all_move_exprs])))
    print("best    {}".format(', '.join(["{} {}".format(m[0], m[1]) for m in best_move_exprs])))
    print("chosen: {} {}, guided {},  prob {:0.2f}, T? {}---loss {:0.4f}".format(
        chos_move, crest, guided, base_prob, correct, loss_data
    ))
    return action_der


def interface_part(num_reads, W):
    partition = [num_reads* W, num_reads, W, 1, W, W, num_reads, 1, 1, num_reads * 3]
    ds = []
    cntr = 0
    for idx in partition:
        tn = [cntr, cntr + idx]
        ds.append(tn)
        cntr += idx
    return ds


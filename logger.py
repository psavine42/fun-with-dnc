# from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from torch.autograd import Variable
from arg import writer

global_step = 0
log_step = 1
# state = ["access_ouptut", "memory", "read_weights", "write_wghts", "link_matrix", "link_weights", "usage"]
interface = ['read_keys', 'read_str', 'write_key', 'write_str',
             'erase_vec', 'write_vec', 'free_gates', 'alloc_gate', 'write_gate', 'read_modes']
losses_desc = ['action', 'ent1-type', 'ent1', 'ent2-type', 'ent2', 'ent3-type', 'ent3']
losses_desc2 = ['action',  'ent1', 'ent2', 'ent3']


def to_log(tnsr):
    if writer:
        clone = tnsr.clone()
        if type(clone) == Variable:
            clone = clone.data
        sizes = list(clone.size())

        if len(sizes) > 1 and not all((s == 1 for s in sizes)):
            return clone.cpu().squeeze().numpy()
        else:
            return clone.cpu().numpy()


def log_if(name, data, f=None):
    if writer:
        if global_step % log_step == 0:
            if f is None:
                writer.add_histogram(name, data, global_step, bins='sturges')
            else:
                f(name, data, global_step)


def log_interface(interface_vec, step):
    if writer:
        for name, data in zip(interface, interface_vec):
            writer.add_histogram("interface." + name, to_log(data), step, bins='sturges')


def log_model(model):
    if writer:
        for name, param in model.named_parameters():
            writer.add_histogram(name, to_log(param), global_step, bins='sturges')


def log_loss(losses, loss):
    if global_step % log_step == 0 and writer:
        type_descs = losses_desc2 if len(losses) == 4 else losses_desc
        for name, param in zip(type_descs, losses):
            writer.add_scalar("lossess." + name, to_log(param), global_step)
        writer.add_scalar('losses.total', loss.clone().cpu().data[0], global_step)


def log_acc(accs, total):
    if global_step % log_step == 0 and writer:
        type_descs = losses_desc2 if len(accs) == 4 else losses_desc
        for name, param in zip(type_descs, accs):
            writer.add_scalar("acc." + name, param, global_step)
        writer.add_scalar('acc.total', total, global_step)

def add_scalar(*args, **kwdargs):
    if writer:
        writer.add_scalar(*args, **kwdargs )


def log_loss_qa(ent, inst, loss):
    if global_step % log_step == 0 and writer:
        writer.add_scalar('losses.ent1-type', ent.clone().cpu().data[0], global_step)
        writer.add_scalar('losses.ent1', inst.clone().cpu().data[0], global_step)
        writer.add_scalar('losses.total', loss.clone().cpu().data[0], global_step)


def log_state(state):
    if global_step % log_step == 0 and len(state) == 8 and writer:
        out, mem, r_wghts, w_wghts, links, l_wghts, usage, hidden = state
        writer.add_histogram("state.access_ouptut", to_log(out), global_step, bins='sturges')
        writer.add_histogram("state.memory", to_log(mem), global_step, bins='sturges')
        writer.add_histogram("state.read_weights", to_log(r_wghts), global_step, bins='sturges')
        writer.add_histogram("state.write_wghts", to_log(w_wghts), global_step, bins='sturges')
        writer.add_histogram("state.link_matrix", to_log(links), global_step, bins='sturges')
        writer.add_histogram("state.link_weights", to_log(l_wghts), global_step, bins='sturges')
        writer.add_histogram("state.usage", to_log(usage), global_step, bins='sturges')

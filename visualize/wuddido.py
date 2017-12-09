import matplotlib.pyplot as plt
import matplotlib.figure as fig
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import losses as L
from utils import depackage, _variable
import copy
from itertools import accumulate
from torch.autograd import Variable
import plotly.plotly as py
import plotly.graph_objs as go
from matplotlib.ticker import FuncFormatter, MaxNLocator

"""
During the state phase, the network must learn to write each tuple to seperate mem location
During the goal phase, the goal phase, the goal tuples are recorded.
--todo-- the triple stored at each location can be recovered by logistic reg. decoder 
--todo-- 
    During state phase read modes - 
    
Extended Data Figure 2
"""

c_map = plt.get_cmap('jet')


def ix_to_color(ix_vec):
    """

    :param ix_vec:
    :return: Image
    """
    ix_np = ix_vec.numpy()


def recorded_step(data, DNC):
    """
            state phase     answer phase
        m   w               r
        e       w               r
        m           w               r
                        w               r
        l
        o
        c
           ------------time------------->
    :param state:
    :return:
    """
    criterion = nn.CrossEntropyLoss()

    phase_masks = data.make_new_problem()
    state = DNC.init_state(grad=False)
    lstm_state = DNC.init_rnn(grad=False)

    prev_action, time_steps = None, []
    # time_steps.append(copy.copy(state) + [-1])

    for phase_idx in phase_masks:
        if phase_idx == 0 or phase_idx == 1:
            inputs = Variable(data.getitem_combined())
            logits, state, lstm_state = DNC(inputs, lstm_state, state)
            #
            _, prev_action = data.strip_ix_mask(logits)
            time_steps.append({'state': copy.copy(state), 'input': inputs.data,
                               'outputs': logits.data, 'phase': phase_idx})

        elif phase_idx == 2:
            inputs = torch.cat([Variable(data.getmask()), prev_action], 1)
            logits, dnc_state, lstm_state = DNC(inputs, lstm_state, state)
            #
            _, prev_action = data.strip_ix_mask(logits)
            time_steps.append({'state': copy.copy(state), 'input': inputs.data,
                               'outputs': logits.data, 'phase': phase_idx})
        else:
            _, all_actions = data.get_actions(mode='both')
            if data.goals_idx == {}:
                break
            mask, pr = data.getmask(), depackage(prev_action)
            final_inputs = Variable(torch.cat([mask, pr], 1))
            #
            logits, state, lstm_state = DNC(final_inputs, lstm_state, state)
            exp_logits = data.ix_input_to_ixs(logits)
            # time_steps.append(copy.copy(state) + [phase_idx])
            final_action, _ = L.naive_loss(exp_logits, all_actions, criterion)

            print(final_action)
            # send action to Data Generator, and set locally
            data.send_action(final_action)
            prev_action = data.vec_to_ix(final_action)
            time_steps.append({'state': copy.copy(state), 'input': final_inputs.data,
                               'outputs': logits.data, 'phase': phase_idx})
    return time_steps


def make_usage_viz(states):
    """

    :param states:
    :return:
    """
    # access, memory, read_wghts, write_wghts, \
    # link, link_wghts, usage, phase_idx = state

    mem_size = list(states[0]['state'][1].size())[1]
    map, writes = [], []
    phases = [c['phase'] for c in states]

    for idx, state in enumerate(states):

        # usage Vectors to calculate write positions
        current_usage = state['state'][6][0].data
        prev_usage = states[idx-1]['state'][6][0].data if idx > 0 else current_usage

        # max diff in usages =>
        diffs = current_usage - prev_usage
        diff_idx, w = max(enumerate(diffs))

        # decode max position
        map.append(np.abs(diffs.numpy()))
        write_vec = state['state'][1][0][diff_idx].data.numpy()
        print(write_vec)

    counts = dict((x, phases.count(x)) for x in set(phases))
    pos = [0] + list(accumulate(counts.values()))[:-1]
    ax = plt.gca()

    # fig = matplotlib.figure(figsize=(3, 3))
    rotated = np.rot90(np.asarray(map))

    ax.xaxis.set_ticks(pos)
    ax.xaxis.set_ticklabels(list(range(4)))

    plot = plt.imshow(rotated, cmap='hot', interpolation='nearest',
                      extent=[0, len(states), 0, mem_size])

    plt.colorbar()
    plt.show()

    pass





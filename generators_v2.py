import os, sys
from torch.utils.data import Dataset
import numpy as np
from timeit import default_timer as timer
import torch
from utils import flat

import problem

import problem.my_air_cargo_problems as mac
from problem.search import *
from problem.planning import Action
from problem.logic import PropKB
from problem.lp_utils import (
    FluentState, encode_state, decode_state,
)
from problem.utils import Expr
import itertools, random


#actions = {'Fly':0, 'Load':1, 'Unload':2}
# actions_1h = {0:'Fly', 1:'Load', 2:'Unload'}

EXPRESSIONS = ['Fly', 'Load', 'Unload', 'At', 'In']
PHASES = ['State', 'Goal', 'Plan', 'Solve']

phase_to_ix = {word: i for i, word in enumerate(PHASES)}
ix_to_phase = {i: word for i, word in enumerate(PHASES)}
exprs_to_ix = {exxp: i for i, exxp in enumerate(EXPRESSIONS)}
ix_to_exprs = {i: exxp for i, exxp in enumerate(EXPRESSIONS)}


default_encoding = {'expr', 'action', 'ent1-type', 'ent1'}
ix_size = [9, 9, 9]
ix2_size = [3, 6, 3, 6, 3, 6]
split1 = [[0, 1], [0, 2], [0, 3]]
split2 = [[0, 1], [1, 2], [0, 3], [1, 4], [0, 5], [1, 6]]

permutations = {'At': {'insert': [0, 0], 'tst': [0, 'P'], 'idx': [2, 2], 'permute': [4, 5, 0, 1, 2, 3]},
                'Fly': {'insert': [0, 0], 'idx':0, 'permute': [0, 1, 2, 3, 4, 5,]},
                'Load': {'insert': [0, 0], 'idx':0, 'permute': [0, 1, 2, 3, 4, 5,]},
                'In': {'insert': [0, 0], 'idx':0, 'permute': [0, 1, 2, 3, 4, 5,]},
                'Unload': {'insert': [0, 0], 'idx':0, 'permute': [0, 1, 2, 3, 4, 5,]},
                }

perm_one = {'At': {'insert': [0], 'tst': [0, 'P'], 'idx': [2, 2], 'permute': [0, 1, 2]},
            'Fly': {'insert': [0], 'idx':0, 'permute': [0, 1, 2]},
            'Load': {'insert': [0], 'idx':0, 'permute': [0, 1, 2]},
            'In': {'insert': [0], 'idx':0, 'permute': [0, 1, 2]},
            'Unload': {'insert': [0], 'idx':0, 'permute': [0, 1, 2]},
                }


class BitData():
    def __init__(self,
                  num_bits=6,
                  batch_size=1,
                  min_length=1,
                  max_length=1,
                  min_repeats=1,
                  max_repeats=2,
                  norm_max=10,
                  log_prob_in_bits=False,
                  time_average_cost=False,
                  name='repeat_copy',):
        self._batch_size = batch_size
        self._num_bits = num_bits
        self._min_length = min_length
        self._max_length = max_length
        self._min_repeats = min_repeats
        self._max_repeats = max_repeats
        self._norm_max = norm_max
        self._log_prob_in_bits = log_prob_in_bits
        self._time_average_cost = time_average_cost

    def _normilize(self, val):
        return val / self._norm_max

    def _unnormalize(self, val):
        return val * self._norm_max

    @property
    def time_average_cost(self):
        return self._time_average_cost

    @property
    def log_prob_in_bits(self):
        return self._log_prob_in_bits

    @property
    def num_bits(self):
        """The dimensionality of each random binary vector in a pattern."""
        return self._num_bits

    @property
    def target_size(self):
        """The dimensionality of the target tensor."""
        return self._num_bits + 1

    @property
    def batch_size(self):
        return self._batch_size

    def get_item(self):
        min_length, max_length = self._min_length, self._max_length
        min_reps, max_reps = self._min_repeats, self._max_repeats
        num_bits = self.num_bits
        batch_size = self.batch_size

        # We reserve one dimension for the num-repeats and one for the start-marker.
        full_obs_size = num_bits + 2
        # We reserve one target dimension for the end-marker.
        full_targ_size = num_bits + 1
        start_end_flag_idx = full_obs_size - 2
        num_repeats_channel_idx = full_obs_size - 1

        # Samples each batch index's sequence length and the number of repeats
        sub_seq_length_batch = torch.from_numpy(
            np.random.randint(min_length, max_length + 1, batch_size)).long()
        num_repeats_batch = torch.from_numpy(
            np.random.randint(min_reps, max_reps + 1, batch_size)).long()

        # Pads all the batches to have the same total sequence length.
        total_length_batch = sub_seq_length_batch * (num_repeats_batch + 1) + 3
        max_length_batch = total_length_batch.max()
        residual_length_batch = max_length_batch - total_length_batch

        obs_batch_shape = [max_length_batch, batch_size, full_obs_size]
        targ_batch_shape = [max_length_batch, batch_size, full_targ_size]
        mask_batch_trans_shape = [batch_size, max_length_batch]

        obs_tensors = []
        targ_tensors = []
        mask_tensors = []

        # Generates patterns for each batch element independently.
        for batch_index in range(batch_size):
            sub_seq_len = sub_seq_length_batch[batch_index]
            num_reps = num_repeats_batch[batch_index]

            # The observation pattern is a sequence of random binary vectors.
            obs_pattern_shape = [sub_seq_len, num_bits]
            # obs_pattern = torch.LongTensor(obs_pattern_shape).uniform_(0, 2).float()
            obs_pattern = torch.from_numpy(np.random.randint(0, 2, obs_pattern_shape)).float()
            print(obs_pattern.size())
            # The target pattern is the observation pattern repeated n times.
            # Some reshaping is required to accomplish the tiling.
            targ_pattern_shape = [sub_seq_len * num_reps, num_bits]
            flat_obs_pattern = obs_pattern.view(-1)
            print(flat_obs_pattern, targ_pattern_shape)
            flat_targ_pattern = flat_obs_pattern.expand(num_reps)
            targ_pattern = torch.reshape(flat_targ_pattern, targ_pattern_shape)

            # Expand the obs_pattern to have two extra channels for flags.
            # Concatenate start flag and num_reps flag to the sequence.
            obs_flag_channel_pad = torch.zeros([sub_seq_len, 2])
            obs_start_flag = torch.eye([start_end_flag_idx], full_obs_size).float()
            num_reps_flag = torch.eye(num_repeats_channel_idx, full_obs_size)

            # note the concatenation dimensions.
            obs = torch.cat([obs_pattern, obs_flag_channel_pad], 1)
            obs = torch.cat([obs_start_flag, obs], 0)
            obs = torch.cat([obs, num_reps_flag], 0)

            # Now do the same for the targ_pattern (it only has one extra channel).
            targ_flag_channel_pad = torch.zeros([sub_seq_len * num_reps, 1])
            targ_end_flag = torch.eye(start_end_flag_idx, full_targ_size)
            targ = torch.cat([targ_pattern, targ_flag_channel_pad], 1)
            targ = torch.cat([targ, targ_end_flag], 0)

            # Concatenate zeros at end of obs and begining of targ.
            # This aligns them s.t. the target begins as soon as the obs ends.
            obs_end_pad = torch.zeros([sub_seq_len * num_reps + 1, full_obs_size])
            targ_start_pad = torch.zeros([sub_seq_len + 2, full_targ_size])

            # The mask is zero during the obs and one during the targ.
            mask_off = torch.zeros([sub_seq_len + 2])
            mask_on = torch.ones([sub_seq_len * num_reps + 1])

            obs = torch.cat([obs, obs_end_pad], 0)
            targ = torch.cat([targ_start_pad, targ], 0)
            mask = torch.cat([mask_off, mask_on], 0)

            obs_tensors.append(obs)
            targ_tensors.append(targ)
            mask_tensors.append(mask)

        # End the loop over batch index.
        # Compute how much zero padding is needed to make tensors sequences
        # the same length for all batch elements.
        residual_obs_pad = [
            torch.zeros(residual_length_batch[i], full_obs_size) for i in range(batch_size)]
        residual_targ_pad = [
            torch.zeros(residual_length_batch[i], full_targ_size) for i in range(batch_size)]

        residual_mask_pad = [torch.zeros(residual_length_batch[i]) for i in range(batch_size)]

        # Concatenate the pad to each batch element.
        obs_tensors = [
            torch.cat([o, p], 0) for o, p in zip(obs_tensors, residual_obs_pad)
            ]
        targ_tensors = [
            torch.cat([t, p], 0) for t, p in zip(targ_tensors, residual_targ_pad)
            ]
        mask_tensors = [
            torch.cat([m, p], 0) for m, p in zip(mask_tensors, residual_mask_pad)
            ]

        # Concatenate each batch element into a single tensor.
        obs = torch.cat(obs_tensors, 1).view(obs_batch_shape)
        targ = torch.cat(targ_tensors, 1).view(targ_batch_shape)
        mask = torch.cat(mask_tensors, 0).reshape(mask_batch_trans_shape).t()
        return obs, targ, mask


class AirCargoData():
    """
        Flags for
        """
    def __init__(self, num_plane=10, num_cargo=6, batch_size=6,
                 num_airport=1000, plan_phase=1,
                 one_hot_size=10, encoding=2, mapping=None,
                 search_function=astar_search, solve=True):
        self.n_plane, self.n_cargo, self.n_airport = num_plane, num_cargo, num_airport
        self.plan_len = plan_phase
        self.batch_size = batch_size
        self.mapping = mapping
        self.solve = solve
        self.encoding = encoding
        self.one_hot_size = [one_hot_size] if type(one_hot_size) == int else one_hot_size
        self.search_fn = search_function

        self.search_param = 'h_ignore_preconditions'
        self.ents_to_ix, self.ix_to_ents = None, None

        self.STATE = ''
        self.current_index = 0
        self.current_problem, self.goals, self.state = None, None, None

        self.phase_oh = torch.eye(len(PHASES))
        self.blnk_vec = torch.zeros(len(EXPRESSIONS) * 2).float() #
        self.expr_o_h = torch.cat([torch.zeros([1, len(EXPRESSIONS)]), torch.eye(len(EXPRESSIONS))], 0).float()

        self.ents_o_h = None
        self.masks = []

        self.cache, self.encodings = {}, {}
        self.make_new_problem()
        print(self.plan_len)

    @property
    def nn_in_size(self):
        return self.blnk_vec.size(-1) + self.phase_oh.size(-1)

    @property
    def nn_out_size(self):
        return self.blnk_vec.size(-1)

    def lookup_expr_to_ix(self, _expr):
        return self.ents_to_ix[str(_expr)]

    def lookup_ix_to_expr(self, ix) -> str:
        if ix in self.ix_to_ents:
            return self.ix_to_ents[ix]
        else:
            return "NA"

    def masked_input(self):

        state_expr = random.choice(self.pull_state())
        state_vec = self.expr_to_vec(state_expr)
        mask_idx = 2
        mask_chunk = state_vec[mask_idx]

        zeros = 0 if type(mask_chunk) == int else tuple([0] * len(mask_chunk))
        masked_state_vec = state_vec.copy()
        masked_state_vec[2] = zeros
        inputs = torch.cat([self.phase_oh[3].unsqueeze(0), self.vec_to_ix(masked_state_vec)], 1)
        return inputs, mask_chunk, state_vec

    def generate_encodings(self):
        """

        :param ix_to_ents:
        :return:
        """
        noops = [torch.zeros(1, i) for i in self.one_hot_size]
        self.ents_o_h = []
        for noop, ix_size in zip(noops, self.one_hot_size):
            self.ents_o_h.append(torch.cat([noop, torch.eye(ix_size)], 0).float())
        self.blnk_vec = torch.cat([torch.zeros([1, len(EXPRESSIONS)]), torch.cat(noops, 1)], 1)

    def print_solution(self, node):
        for action in node.solution():
            print("{}{}".format(action.name, action.args))

    def best_logic(self, action_exprs):
        #index of cargo : airport
        goals = {goal.args[0]: goal.args[1] for goal in self.goals}
        best_actions = []
        for action in action_exprs:
            op = action.op
            if sym == 'Fly':
                cargo = action.args[0]
                # if there is a cargo in the plane
                # and desitination is the cargo's goal

                # if there are no cargos at the airport

            elif sym == 'Unload':
                cargo = action.args[0]
                if goals[cargo] == action.args[2]:
                    best_actions.append(action)
            elif sym == 'Load':
                # load cargo if it is not in its home
                cargo = action.args[0]
                if goals[cargo] != action.args[2]:
                    best_actions.append(action)
        return best_actions

    def get_actions(self, mode='best'):
        self.current_problem.initial = self.STATE
        if mode == 'all':
            actions_exprs = self.current_problem.actions(self.STATE)
        elif mode == 'one':
            actions_exprs = [self.current_problem.one_action(self.STATE)]
        elif mode == 'both':
            pass
        else:
            #state_hash = hash(str(self.pull_state()))
            #if state_hash in self.cache:
                #actions_exprs = self.cache[state_hash]
            #else:
            if self.search_param is not None:
                prm = getattr(self.current_problem, self.search_param)
                solution = self.search_fn(self.current_problem, prm)
                actions_exprs = solution.solution()[0:len(self.current_problem.planes)]
            else:
                solution = self.search_fn(self.current_problem)
                actions_exprs = solution.solution()[0:len(self.current_problem.planes)]
             #  self.cache[state_hash] = actions_exprs
             
        return [self.expr_to_vec(a) for a in actions_exprs]


    def encode_action(self, action_obj):
        return torch.from_numpy(self.current_problem.encode_action(action_obj)).long()

    def send_action(self, action_vec):
        self.current_problem.initial = self.STATE
        sym, args = self.vec_to_expr(action_vec)

        actions_ = self.current_problem.actions(self.STATE)
        final_act = []
        for a in actions_:
            if a.name == sym and all((str(ar) == at) for ar, at in zip(a.args, args)):
                final_act = a
                break
        assert final_act != []
        self.STATE = self.current_problem.result(self.STATE, final_act)
        return self.STATE, final_act

    def vec_to_expr(self, _vec):
        action_str = EXPRESSIONS[_vec[0]]
        args_vec = []
        for idx, value in enumerate(_vec[1:]):
            args_vec.append(self.lookup_ix_to_expr(value))
        return action_str, args_vec

    def pull_state(self):
        res = []
        for idx, char in enumerate(self.STATE):
            if char == 'T':
                res.append(self.current_problem.state_map[idx])
        return res

    def vec_to_ix(self, _vec):
        """
        Input target vec representing cross entropy loss target  [1 0 2 0 0 0 0]
            Returns a one hot version of it as training input       [01 00, 100, 000, 000, 000]
        :param _vec:
        :return:
        """
        merged = flat(_vec)
        action = self.expr_o_h[merged[0]].unsqueeze(0)
        ix_ent = []
        merged = merged[1:]
        if self.mapping is not None:
            expr_str = EXPRESSIONS[_vec[0]]
            # print(expr_str)
            mp = self.mapping[expr_str]
            permute = mp['permute'] if 'permute' in mp else None
            insert_ = mp['insert'] if 'insert' in mp else None
            test = mp['tst'] if 'tst' in mp else None
            idx = mp['idx'] if 'idx' in mp else None
            if insert_ is not None and test is not None and idx is not None:
                text_ix = _vec[1:][test[0]]
                for i, insrt_ent in zip(idx, insert_):
                    if test[1] in self.lookup_ix_to_expr(text_ix):
                        merged.insert(i, insrt_ent)
                    else:
                        merged.append(insrt_ent)
            elif insert_ is not None and idx is None:
                for insrt_ent in insert_:
                    merged.append(insrt_ent)
            #print('merged2', merged)
            if permute is not None:
                nperm = np.argsort(permute)
                merged = np.asarray(merged)[nperm]
        else:
            # print(len(self.ents_o_h))
            for idx in range(len(self.ents_o_h) - len(merged)):
                merged.append(0)

        for idx, value in enumerate(merged):
            ix_ent.append(self.ents_o_h[idx][value].unsqueeze(0))
        return torch.cat([action, torch.cat(ix_ent, 1)], 1)

    def ix_to_vec(self, ix_ent):
        action_l = self.expr_o_h.size(-1)
        ent_vec = [ix_ent[0:action_l].index(1) + 1]
        start = action_l
        for idx, ix_size in enumerate(self.one_hot_size):
            expr_ix = ix_ent[start:start+ix_size]
            if 1 in expr_ix:
                ent_vec.append(expr_ix.index(1) + 1)
            else:
                ent_vec.append(0)
            start += ix_size
        return ent_vec

    def ix_to_ixs(self, ix_ent, grouping=None):
        action_l = self.expr_o_h.size(-1)
        ixs_vec = [ix_ent[:, 0:action_l]]
        start = action_l
        if grouping is None:
            grouping = self.one_hot_size
        for idx, ix_size in enumerate(grouping):
            ixs_vec.append(ix_ent[:, start:start + ix_size])
            start += ix_size
        return ixs_vec

    def strip_ix_mask(self, ix_input_vec):
        phase_size = self.phase_oh.size(-1)
        ixs_vec = ix_input_vec[:, phase_size:]
        phase_vec = ix_input_vec[:, :phase_size]
        return phase_vec, ixs_vec

    def ix_input_to_ixs(self, ix_input_vec, grouping=None):
        """

        :param ix_input_vec:
        :param grouping:
        :return:
        """
        phase_size = self.phase_oh.size(-1)
        ixs_vec = ix_input_vec[:, phase_size:]
        return self.ix_to_ixs(ixs_vec, grouping)

    def ix_to_expr(self, ix_input_vec):
        vec = self.ix_to_vec(ix_input_vec)
        return self.vec_to_expr(vec)


    def expr_to_vec(self, expr_obj):
        """
        :param expr_obj: Action Expr object Fly(P0, A1, A2)
        :return: action vec [0, 0, 1, 2], and argument permutation
        """
        if type(expr_obj) == Action:
            exp_name = expr_obj.name
        else:
            exp_name = expr_obj.op
        ent_vec = [exprs_to_ix[exp_name]]
        for arg in expr_obj.args:
            ent_vec.append(self.lookup_expr_to_ix(arg))

        return ent_vec

    def gen_state_vec(self, index):
        state_expr = self.state[index]
        return self.expr_to_vec(state_expr)

    def gen_input_ix(self, _exprs, index):
        """
        Generate a one_hot vector at a given index
        :param vecs:
        :param index:
        :return:
        """
        _expr = _exprs[index]
        ent_vec = self.expr_to_vec(_expr)
        return self.vec_to_ix(ent_vec)

    def human_readable(self, inputs, mask=None) -> str:
        if mask is None:
            phase = PHASES[self.phase_oh[self.masks[self.current_index]]]
        elif mask is False:
            phase = ''
        elif mask is True:
            phase_size = self.phase_oh.size(-1)
            phase_ix = inputs[:, :phase_size].squeeze()
            # phase_ix.

            inputs = inputs[:, phase_size:]
            phase = ''

        elif isinstance(mask, int):
            phase = PHASES[mask]
        else:
            phase = PHASES[mask.squeeze()[0]]
        args = []
        txt = ''
        if phase:
            args.append(phase)
            txt += 'Phase {}, '

        if isinstance(inputs, torch.Tensor):
            expr = self.ix_to_expr(inputs)
        else:
            expr = self.vec_to_expr(vec)
        args.append(expr)
        txt += 'expr {}'
        return txt.format(args)


    def make_new_problem(self):
        """
            Set up new problem object
        :return:
        """
        problem, (e_ix, ix_e), (s, g) = mac.arbitrary_ACP2(self.n_airport, self.n_cargo,
                                                           self.n_plane, encoding=self.encoding)
        self.current_problem = problem
        self.STATE = problem.initial
        self.generate_encodings()
        self.state = s
        self.goals = g
        if self.solve is True:
            prm = getattr(problem, self.search_param)
            solution_node = len(self.search_fn(problem, prm).solution())
        else:
            solution_node = self.solve

        state = torch.zeros(len(self.state))
        goal = torch.ones(len(self.goals))
        plan = torch.ones(self.plan_len) * 2
        resp = torch.ones(solution_node + 3) * 3

        self.masks = torch.cat([state, goal, plan, resp], 0).long()
        self.ents_to_ix,  self.ix_to_ents = e_ix, ix_e
        self.current_index = 0
        return self.masks

    def len__(self):
        if self.current_index >= self.current_problem[0].size(0):
            self.make_new_problem()
        return len(self.current_problem[1])

    def getitem(self, batch=1):
        """Returns a problem, [initial-state, goals]
            and a runnable solution object [problem, solution_node]

            Otherwise take the target one_hot class mask in form of
            [ent1-type, ent1 ....entN, channel]

            """
        if self.current_index >= len(self.masks):
            self.make_new_problem()

        phase = self.masks[self.current_index]
        if phase == 0:
            inputs = self.gen_input_ix(self.state, self.current_index)
        elif phase == 1:
            inputs = self.gen_input_ix(self.goals, self.current_index - len(self.state))
        elif phase == 2:
            inputs = self.blnk_vec
        else:
            inputs = self.blnk_vec

        self.current_index += batch
        return inputs, self.phase_oh[phase].unsqueeze(0)

    def getmask(self, batch=1):
        if self.current_index >= len(self.masks):
            self.make_new_problem()

        phase = self.masks[self.current_index]
        self.current_index += batch
        return self.phase_oh[phase].unsqueeze(0)

    def getitem_combined(self, batch=1):
        inputs, mask = self.getitem(batch)
        return torch.cat([mask, inputs], 1)



import os
import sys
from torch.utils.data import Dataset
import numpy as np
from timeit import default_timer as timer
import torch
parent = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.dirname(parent))
import aimacode
import my_air_cargo_problems as mac
from aimacode.search import *
from aimacode.utils import Expr

props = {'At':0, 'In':1}
actions = {'Fly':0, 'Load':1, 'Unload':2}
actions_1h = {0:'Fly', 1:'Load', 2:'Unload'}

exprs = ['Fly', 'Load', 'Unload', 'At', 'In']

phases = ['State', 'Goal', 'Plan', 'Solve']

phase_to_ix = {word: i for i, word in enumerate(phases)}
ix_to_phase = {i: word for i, word in enumerate(phases)}
exprs_to_ix = {exxp: i for i, exxp in enumerate(exprs)}
ix_to_exprs = {i: exxp for i, exxp in enumerate(exprs)}

def encoding_():
    pass

def swap_fly(fly_action):
    """

    :param fly_action: Fly(P1, A1, A0)
    :return: Fly(P1 _ A0)
    """

    pass

class Encoded_Expr():
    def __init__(self, op, args):
        self.op = str(op)
        self.args = args
        self.one_hot = []

    def vec_to_expr(self):
        pass




class EncodedAirCargoProblem(mac.AirCargoProblem):
    def __init__(self, problem, init_vec, goals_vec, mp_merged, one_hot_size):
        self.problem = problem
        self.succs = self.goal_tests = self.states = 0
        self.found = None
        self.init_state = init_vec

        self.goal_state = goals_vec
        # dictionary of mappings of ents to one_hot
        self.problem_ents = mp_merged
        # dictionary pf mappings of one_hot to ents
        self.ent_to_vec = self.flip_encoding(mp_merged)
        print(self.problem_ents)
        print(self.ent_to_vec)
        self.one_hot_size = one_hot_size
        self.solution_node = None
        self.entity_o_h = torch.eye(one_hot_size).long()
        self.action_o_h = torch.eye(3).long()
        self.types_o_h = torch.eye(3).long()
        self.phases_o_h = torch.LongTensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])

    def flip_encoding(self, ents):
        encoding = {}
        for key, value in ents.items():
            encoding[str(list(value))] = key
        return encoding

    def reverse_lookup(self, one_hot):
        return next(key for key, value in self.problem_ents.items() if value == one_hot)

    def action_expr_to_vec(self, action_expr):
        """

        :param action_expr: Action Expr object Fly(P0, A1, A2)
        :return: action vec [0,
        """
        action_vec = [actions[action_expr.name]]
        for arg in action_expr.args:
            e_type, ent = self.problem_ents[str(arg)]
            action_vec.append(e_type)
            action_vec.append(ent)
        print("action_vec", action_vec)
        return np.asarray(action_vec, dtype=int)

    def get_best_action_vecs(self, solution_node):
        """

        :param solution_node: Graph Node reporesenting solution
        :return: vectors for each expr in solution
        """
        action_vecs = []
        for action_expr in solution_node.solution()[0:len(self.problem.planes)]:
            action_vec = self.action_expr_to_vec(action_expr)
            action_vecs.append(torch.from_numpy(action_vec))
        return action_vecs

    def get_all_actions(self, state):
        zz = [torch.from_numpy(self.action_expr_to_vec(a)) for a in self.problem.actions(state)]
        return zz

    def encode_solution(self, solution):
        zz = [torch.from_numpy(self.action_expr_to_vec(a)) for a in solution]
        return zz

    def decode_action(self, coded_action):
        sym = actions_1h[coded_action[0]]
        ent1 = str(list(coded_action[1:3]))
        ent2 = str(list(coded_action[3:5]))
        ent3 = str(list(coded_action[5:7]))
        ex1 = self.ent_to_vec[ent1]
        ex2 = self.ent_to_vec[ent2]
        ex3 = self.ent_to_vec[ent3]
        return sym, [ex1, ex2, ex3]

    def send_action(self, state, coded_action):
        sym, args = self.decode_action(coded_action)
        actions_ = self.actions(state)
        print(actions_)
        final_act = []
        for a in actions_:
            if a.name == sym and all((str(ar) == at) for ar, at in zip(a.args, args)):
                final_act = a
                break
        assert final_act != []
        print(final_act)
        result_state = self.problem.result(state, final_act)
        return result_state

    def actions(self, state):
        self.succs += 1
        return self.problem.actions(state)

    def result(self, state, action):
        self.states += 1
        return self.problem.result(state, action)

    def goal_test(self, state):
        self.goal_tests += 1
        result = self.problem.goal_test(state)
        if result:
            self.found = state
        return result

    def run_search(self, search_fn, parameter=None):
        if parameter is not None:
            prm = getattr(self.problem, parameter)
            node = search_fn(self.problem, prm)
        else:
            node = search_fn(self.problem)
        return node

    def path_cost(self, c, state1, action, state2):
        return self.problem.path_cost(c, state1, action, state2)

    def value(self, state):
        return self.problem.value(state)

    def __getattr__(self, attr):
        return getattr(self.problem, attr)


class AirCargoData():
    """
        Flags for
        """
    def __init__(self,
                 num_plane=10, num_cargo=6, batch_size=6,
                 num_airport=1000, one_hot_size=10, mode='loose',
                 search_function=astar_search):
        self.num_plane = num_plane
        self.num_cargo = num_cargo
        self.num_airport = num_airport
        self.batch_size = batch_size
        self._actions_mode = mode
        self.one_hot_size = one_hot_size
        self.search_fn = search_function
        self.STATE = ''
        self.search_param = 'h_ignore_preconditions'
        self.problem = None
        self.current_problem = None
        self.current_index = 0
        #self.make_new_problem()

    def print_solution(self, node):
        for action in node.solution():
            print("{}{}".format(action.name, action.args))

    def phase_vec(self, tensor_, add_channel):
        chans = torch.stack([torch.LongTensor(add_channel) for _ in range(tensor_.size(0))], dim=0)
        return torch.cat([chans, tensor_.long()], dim=-1)

    def get_actions(self, mode='strict'):
        self.problem.problem.initial = self.STATE
        if mode == 'all':
            return self.problem.get_all_actions(self.STATE)
        else:
            solution = self.problem.run_search(self.search_fn, self.search_param)
            return self.problem.get_best_action_vecs(solution)

    def encode_action(self, action_obj):
        return torch.from_numpy(self.problem.encode_action(action_obj)).long()

    def send_action(self, coded_action):
        self.problem.problem.initial = self.STATE
        self.STATE = self.problem.send_action(self.STATE, coded_action)
        return True

    def vec_to_one_hot(self, coded_ent):
        ents = []
        for idx in range(1, 7, 2):
            e_type = coded_ent[idx]
            ent = coded_ent[idx + 1]
            if e_type == 0 and ent == 0:
                ents.append(torch.zeros(3 + self.one_hot_size).long())
            else:
                ents.append(torch.cat([self.problem.types_o_h[e_type], self.problem.entity_o_h[ent]], 0))
        return torch.cat(ents, 0)

    def expand_state_vec(self, coded_state):
        """Input target vec representing cross entropy loss target  [1 0 2 0 0 0 0]
            Returns a one hot version of it as training input       [01 00, 100, 000, 000, 000]"""
        ents = self.vec_to_one_hot(coded_state)
        phase = self.problem.phases_o_h[coded_state[0]]
        return torch.cat([phase, ents], 0)

    def expand_action_vec(self, coded_action):
        """Input target vec representing action
            [1 0 2 0 0 0 0]
            Returns a one hot version of it as training input
            [01 00, 100, 000, 000, 000]
            """
        ents = self.vec_to_one_hot(coded_action)
        action = self.problem.action_o_h[coded_action[0]]
        return torch.cat([action, ents], 0).unsqueeze(0).float()

    def make_new_problem(self):
        acp, i, g, m = mac.arbitrary_ACP(self.num_airport, self.num_plane,
                                         self.num_cargo, one_hot_size=self.one_hot_size)
        problem = EncodedAirCargoProblem(acp, i, g, m, self.one_hot_size)

        # print(problem)
        # run the solution to determine how long to give dnc
        solution_node = problem.run_search(self.search_fn, self.search_param)

        word_len = len(problem.init_state[0])
        len_init_phase = len(problem.init_state)
        len_goal_phase = len(problem.goal_state)
        len_plan_phase = 10
        len_resp_phase = len(solution_node.solution()) + 6

        # determine the number of iterations input will happen for
        mask_zero = torch.zeros(len_init_phase + len_goal_phase + len_plan_phase)
        mask_ones = torch.ones(len_resp_phase)
        masks = torch.cat([mask_zero, mask_ones], 0)

        init_phs_data = self.phase_vec(torch.from_numpy(problem.init_state), [0])
        goal_phs_data = self.phase_vec(torch.from_numpy(problem.goal_state), [1])
        # during planning and response phases, there is no inputs.
        plan_phs_data = self.phase_vec(torch.zeros(len_plan_phase, word_len), [2])
        resp_phs_data = self.phase_vec(torch.zeros(len_resp_phase, word_len), [3])

        inputs = torch.cat([init_phs_data, goal_phs_data, plan_phs_data, resp_phs_data], 0)
        self.current_problem = [inputs, masks]
        self.problem = problem
        self.STATE = problem.initial
        self.current_index = 0
        return problem.problem, solution_node, masks

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
        if self.current_index >= self.current_problem[0].size(0):
            self.make_new_problem()

        masks = self.current_problem[1][self.current_index:self.current_index+batch]
        inputs = self.current_problem[0][self.current_index:self.current_index + batch]
        inputs = torch.stack([self.expand_state_vec(i).float() for i in inputs], 0)
        self.current_index += batch

        return inputs, masks


class RandomData(Dataset):
    def __init__(self,
                 num_seq=10,
                 seq_len=6,
                 iters=1000,
                 seq_width=4):
        self.seq_width = seq_width
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.iters = iters

    def __getitem__(self, index):
        con = np.random.randint(0, self.seq_width, size=self.seq_len)
        seq = np.zeros((self.seq_len, self.seq_width))
        seq[np.arange(self.seq_len), con] = 1
        end = torch.from_numpy(np.asarray([[-1] * self.seq_width])).float()
        zer = np.zeros((self.seq_len, self.seq_width))
        return seq, zer

    def __len__(self):
        return self.iters

class GraphData(Dataset):
    def __init__(self,
                 num_seq=10,
                 seq_len=6,
                 iters=1000,
                 domain=None,
                 actions=None,
                 start_state=None,
                 seq_width=4):
        """
            Each vector encoded a triple consisting of a source label,
                an edge label and a destination label.
            All labels were represented as numbers between 0 and 999,
                with each digit represented as a 10-way one-hot encoding.
            We reserved a special ‘blank’ label, represented by the all-zero vector
                for the three digits, to indicate an unspecified label.
            Each label required 30 input elements, and each triple required 90.
            The sequences were divided into multiple phases:
                1) first a graph description phase,
                then a series of query (2 Q) and answer (3 A) phases;
                    in some cases the Q and A were separated by an additional planning phase
                    with no input, during which the network was given time to compute the answer.
            During the graph description phase, the triples defining the input graph were
                presented in random order.
            Target vectors were present during only the answer phases.

            Params from Paper:
            GRAPH
                input vectors were size 92
                    90 info
                    binary chan for phase transition
                    binary chan for when prediction is needed
                target vectors 90
            BABI
                input vector of 159 one-hot-vector of words (156 unique words + 3 tokens)

            Propositions:
            :: one-hot 0-9
            :: [At     ( C1     , SFO    )]
            :: [{0..1} ( {0..1} , {0..1} )] + [True/False, end-exp, pred-required?]
            Load({}, {}, {})
            special tokens [ '(', ')', ',',  ]

            Actions vs Propositions
                    Action    =>  precond_pos  , precond_neg , effects_pos  , effects_ned]
                    Eat(Cake) => [[Have(Cake) ], [],         , [Eaten(Cake)], [Not Have(Cake)]]
                  +---------------------------------------------------------------------------+
                  |  1 9 3  9 8     4   9 2 9  3                 5   9  2 9 3   7    4 9 2  9 |
                  |  1                                                                        |
                  +---------------------------------------------------------------------------+

                  input:          T?  Op   (   Pred   )
                                +----------------------------+
                  start-seq     |                           1|
                  Eat(Cake)     | 00 0001 1001 0011 1001 0  0|     Action Name      | h * h * args
                  Have(Cake)    | 11 0100 1001 0011 1001 1  0|    } Pre-Conditions
                        .       |                        0  0|
                        .       |                        0  0|
                  Eaten(Cake)   | 11 0101 1001 0011 1001 1  0|    } Post-Conditions
                  ¬ Have(Cake)  | 01 0100 1001 0011 1001 0  1|
                                +----------------------------+

                  Final input is concat of all statements

                  Goal -> At(C1, Place )

            Paper (first block, adjacency relation, second block)
                  (100000,      1000,               010000)
                  “block 1      above               block 2”

            let the goals be 1 of 26 possible letters designated by one-hot encodings;
            that is, A =​ (1, 0, ..., 0), Z =​ (0, 0, ..., 1) and so on

            -The board is represented as a set of place-coded representations, one for each square.
                Therefore, (000000, 100000, ...) designates that the bottom,
                left-hand square is empty, block 1 is in the bottom centre square, and so on

            The network also sees a binary flag that represents a ‘go cue’.
            While the go cue is active, a goal is selected from the list of goals that have
                been shown to the network, its label is retransmitted to the network for one
                time-step, and the network can begin to move the blocks on the board.

            All told, the policy observes at each time-step a vector with features

                            Constraints 16
                (goal name, first block, adjacency relation, second block, go cue, board state).
                [26 ...     (6             4                  6      )x6    1       63]  -> 186 ~state

                there are 7 possible actions so output is mapped to size 7 one_hot vector.

                10 goals -> 250,

            Up to 10 goals with 6 constraints each can be sent to the network before action begins.

            Once the go cue arrives, it is possible for the policy network to move a block
                from one column to another or to pass at each turn.
            We parameterize these actions using another one-hot encoding so that,
                for a 3 ×​ 3 board, a move can be made from any column to any other;
                with the pass move, there are therefore 7 moves.


            8 Airports  , 4 Cargos, 4 Airplanes
            0001        , 0000    , 0100

            Network - [2 x 250]



            Input at t: prev


            Types of tasks ->
                1) given True Statements, we may want to generate negative statements
                2)

            """
        self.seq_width = seq_width
        self.num_seq = num_seq
        self.seq_len = seq_len
        self.iters = iters

    def __getitem__(self, index):
        #con = np.random.randint(0, self.seq_width, size=self.seq_len)



        return None #seq, zer

    def __len__(self):
        return self.iters


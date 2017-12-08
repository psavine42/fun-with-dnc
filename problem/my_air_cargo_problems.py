

from .logic import PropKB
from .planning import Action
from .search import (
    Node, Problem
)
from .utils import expr
from .lp_utils import (
    FluentState, encode_state, decode_state,
)
from .my_planning_graph import PlanningGraph
from functools import lru_cache
from random import shuffle


class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
        """
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        """
        def load_actions():
            """Create all concrete Load actions and return a list

            :return: list of Action objects
            """
            loads = []
            for p in self.planes:
                for a in self.airports:
                    for c in self.cargos:
                        precond_pos = [expr("At({}, {})".format(c, a)),
                                       expr("At({}, {})".format(p, a))]
                        # remove this for tests where plane can load more than cargo
                        precond_neg = [expr("In({}, {})".format(c1, p)) for c1 in self.cargos]
                        effect_add = [expr("In({}, {})".format(c, p))]
                        effect_rem = [expr("At({}, {})".format(c, a))]
                        act = Action(expr("Load({}, {}, {})".format(c, p, a)),
                                     [precond_pos, precond_neg],
                                     [effect_add, effect_rem])
                        loads.append(act)

            return loads

        def unload_actions():
            """Create all concrete Unload actions and return a list

            :return: list of Action objects
            """
            unloads = []
            # create all Unload ground actions from the domain Unload action
            for p in self.planes:
                for a in self.airports:
                    for c in self.cargos:
                        precond_pos = [expr("In({}, {})".format(c, p)),
                                       expr("At({}, {})".format(p, a))]

                        effect_add = [expr("At({}, {})".format(c, a))]
                        effect_rem = [expr("In({}, {})".format(c, p))]
                        act = Action(expr("Unload({}, {}, {})".format(c, p, a)),
                                     [precond_pos, []],
                                     [effect_add, effect_rem])
                        unloads.append(act)

            return unloads

        def fly_actions():
            """Create all concrete Fly actions and return a list

            :return: list of Action objects
            """
            flys = []

            for fr in self.airports:
                for to in self.airports:
                    if fr != to:
                        for p in self.planes:
                            precond_pos = [expr("At({}, {})".format(p, fr))]
                            # precond_neg = []
                            effect_add = [expr("At({}, {})".format(p, to))]
                            effect_rem = [expr("At({}, {})".format(p, fr))]
                            fly = Action(expr("Fly({}, {}, {})".format(p, fr, to)),
                                         [precond_pos, []],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys
        data = load_actions() + unload_actions() + fly_actions()
        shuffle(data)
        return data

    def one_action(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

            :param state: str
                state represented as T/F string of mapped fluents (state variables)
                e.g. 'FTTTFF'
            :return: list of Action objects
            """
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())

        for action in self.actions_list:
            is_possible = True
            for clause in action.precond_pos:
                if clause not in kb.clauses:
                    is_possible = False
                    break
            for clause in action.precond_neg:
                if clause in kb.clauses:
                    is_possible = False
                    break
            if is_possible:
                return action
        return []

    def actions(self, state: str) -> list:
        possible_actions = []
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for action in self.actions_list:
            is_possible = True
            for clause in action.precond_pos:
                if clause not in kb.clauses:
                    is_possible = False
                    break
            for clause in action.precond_neg:
                if clause in kb.clauses:
                    is_possible = False
                    break
            if is_possible:
                possible_actions.append(action)
        return possible_actions

    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given
            action in the given state. The action must be one of
            self.actions(state).

            :param state: state entering node
            :param action: Action applied
            :return: resulting state after action
            """
        new_state = FluentState([], [])
        old_state = decode_state(state, self.state_map)

        for fluent in old_state.pos:
            if fluent not in action.effect_rem:
                new_state.pos.append(fluent)

        for fluent in action.effect_add:
            if fluent not in new_state.pos:
                new_state.pos.append(fluent)

        for fluent in old_state.neg:
            if fluent not in action.effect_add:
                new_state.neg.append(fluent)

        for fluent in action.effect_rem:
            if fluent not in new_state.neg:
                new_state.neg.append(fluent)
        return encode_state(new_state, self.state_map)

    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

            :param state: str representing state
            :return: bool
            """
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    @lru_cache(maxsize=8192)
    def h_pg_levelsum(self, node: Node):
        """This heuristic uses a planning graph representation of the problem
            state space to estimate the sum of all actions that must be carried
            out from the current state in order to satisfy each individual goal
            condition.
            """
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    @lru_cache(maxsize=8192)
    def h_ignore_preconditions(self, node: Node):
        """This heuristic estimates the minimum number of actions that must be
            carried out from the current state in order to satisfy all of the goal
            conditions by ignoring the preconditions required for an action to be
            executed.
            """
        kb = PropKB()
        kb.tell(decode_state(node.state, self.state_map).pos_sentence())
        return len([c for c in self.goal if c not in kb.clauses])


def posneg_helper(cargos, planes, airports, pos_exprs):
    """given the positive expressions and entities, generate the negative expressions"""
    pos_set = set(pos_exprs)
    neg = []
    for c in cargos:
        for p in planes:
            neg.append(expr("In({}, {})".format(c, p)))
        for a in airports:
            neg.append(expr("At({}, {})".format(c, a)))
    for a in airports:
        for p in planes:
            neg.append(expr("At({}, {})".format(p, a)))
    return list(set(neg).difference(pos_set))

#def ¬(x): 
#   print(x) well, python does not allow arbitrary symbols is guess. 
    #pass

def air_cargo_p1() -> AirCargoProblem:
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)')]
    neg = posneg_helper(cargos, planes, airports, pos)
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)')]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p2() -> AirCargoProblem:
    cargos = ['C1', 'C2', 'C3']
    planes = ['P1', 'P2', 'P3']
    airports = ['JFK', 'SFO', 'ATL']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           expr('At(P3, ATL)')]
    neg = posneg_helper(cargos, planes, airports, pos)
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, SFO)')]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p3() -> AirCargoProblem:
    cargos = ['C1', 'C2', 'C3', 'C4']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO', 'ATL', 'ORD']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(C4, ORD)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)')]
    neg = posneg_helper(cargos, planes, airports, pos)
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, JFK)'),
            expr('At(C4, SFO)')]
    return AirCargoProblem(cargos, planes, airports, init, goal)

#####################################################################
###                         GENERATORS
#####################################################################
# this is all for generating DNC data

import numpy as np
import random
from random import randint
random.seed()


def bin_gen(num, one_hot_size):
    """In case I need to do a binary encoding vs one_hot
        """
    int(bin(num)[2:].zfill(one_hot_size))



def air_cargo_generator_v1(num_plane, num_airport, num_cargo,
                           one_hot_size=None, vec_fn=None):
    """Init:
        each plane must be at airport
        each cargo must be at an airport
        each cargo has a destination

        Returned tuples of <A, P, C> as one hot vecs of equal size
        zeros vector represents nothing at that slot.
        so 'At(C1, SFO)' with one_hot_size = 4 ->
         S, Act, A, P, C
        +---------------+
        | 1-0  2-0  3-0 |
        +---------------+

        'state' size becomes max A * P * 1h + A * C * 1h (4, 4, 4, 10)
        """
    if one_hot_size is None:
        one_hot_size = max([num_airport, num_cargo, num_plane]) + 3
    if vec_fn is None:
        vec_fn = np.eye

    assert num_airport <= one_hot_size
    encoding_vec = vec_fn(one_hot_size)

    o_h_planes = encoding_vec[:num_plane].astype(int)
    o_h_airprt = encoding_vec[:num_airport].astype(int)
    o_h_cargos = encoding_vec[:num_cargo].astype(int)
    o_h_types = vec_fn(3).astype(int)

    zeros = np.zeros(one_hot_size, dtype=int)
    zero_type = np.zeros(3, dtype=int)
    zero_vec = np.concatenate([zero_type, zeros], axis=0)

    o_h_state, o_h_goals = [], []
    idx_state, idx_goals = [], []
    airprt_dict, cargos_dict, planes_dict = {}, {}, {}

    for idx, o_h_airpr in enumerate(o_h_airprt):
        o_h_airpr = np.concatenate([o_h_types[0], o_h_airpr], axis=0)
        airprt_dict[str(o_h_airpr)] = 'A{}'.format(idx)

    #planes can go wherever
    for idx, o_h_plane in enumerate(o_h_planes):
        #generate random
        airprt_idx = randint(0, num_airport - 1)
        #make one_hot vecs
        airprt_vec = np.concatenate([o_h_types[0], o_h_airprt[airprt_idx]], axis=0)
        plane_vec = np.concatenate([o_h_types[1], o_h_plane], axis=0)
        #make final state vecs and add to states.
        oh_plane = np.concatenate([airprt_vec, plane_vec, zero_vec], axis=0)
        o_h_state.append(oh_plane)
        #lookup idxs
        pl_ent = 'P{}'.format(idx)
        planes_dict[str(plane_vec)] = pl_ent
        #make state dicts and add to states.
        ar_ent = airprt_dict[str(airprt_vec)]
        idx_state.append('At({}, {})'.format(pl_ent, ar_ent))

    # cargos start end end are mutually exclusive
    # for porpises of this problem. Erp erp.
    for idx, o_h_cargo in enumerate(o_h_cargos):
        # generate random
        init_idx = random.randint(0, num_airport-1)
        allowed_values = set(range(num_airport))
        allowed_values.discard(init_idx)
        goal_idx = random.choice(list(allowed_values))
        # make one_hot vecs
        air_in_vec = np.concatenate([o_h_types[0], o_h_airprt[init_idx]], axis=0)
        air_gl_vec = np.concatenate([o_h_types[0], o_h_airprt[goal_idx]], axis=0)
        cargo_vec = np.concatenate([o_h_types[2], o_h_cargo], axis=0)
        # make final state vecs and add to states.
        oh_init = np.concatenate([air_in_vec, zero_vec, cargo_vec], axis=0)
        oh_goal = np.concatenate([air_gl_vec, zero_vec, cargo_vec], axis=0)
        o_h_state.append(oh_init)
        o_h_goals.append(oh_goal)
        # lookup idxs
        cr_ent = 'C{}'.format(idx)
        cargos_dict[str(cargo_vec)] = cr_ent
        state_ar_ent = airprt_dict[str(air_in_vec)]
        goals_ar_ent = airprt_dict[str(air_gl_vec)]
        # make state dicts and add to states.
        idx_state.append('At({}, {})'.format(cr_ent, state_ar_ent))
        idx_goals.append('At({}, {})'.format(cr_ent, goals_ar_ent))

    return [[np.asarray(o_h_state), np.asarray(o_h_goals)],
            [idx_state, idx_goals],
            [airprt_dict, cargos_dict, planes_dict]]

def reverse_lookup(problem_ents, o_h_idx):
    key = next(key for key, value in problem_ents.items() if np.array_equal(value, o_h_idx))
    return key

noop = np.asarray([0, 0], dtype=int)


def air_cargo_generator_v2(num_airport, num_cargo, num_plane,
                           one_hot_size=None, vec_fn=None):
    """Init:
        each plane must be at airport
        each cargo must be at an airport
        each cargo has a destination

        Returned typed state expressions with type and instance of each object
        0-0 indicates 
          A     P    C
        +---------------+
        | 1-0  2-0  3-0 |
        +---------------+
        """
    if one_hot_size is None:
        one_hot_size = max([num_airport, num_cargo, num_plane]) + 3
    if vec_fn is None:
        vec_fn = np.eye

    assert num_airport <= one_hot_size
    print(num_airport)
    o_h_airprt = np.asarray([[0, i] for i in range(num_airport)], dtype=int)
    o_h_planes = np.asarray([[1, i] for i in range(num_plane)], dtype=int)
    o_h_cargos = np.asarray([[2, i] for i in range(num_cargo)], dtype=int)

    o_h_state, o_h_goals = [], []
    idx_state, idx_goals = [], []
    airprt_dict, cargos_dict, planes_dict = {}, {}, {}

    for idx, o_h_airpr in enumerate(o_h_airprt):
        airprt_dict['A{}'.format(idx)] = o_h_airpr

    # planes can go wherever
    for idx, o_h_plane in enumerate(o_h_planes):
        # generate random
        airprt_idx = randint(0, num_airport - 1)
        # make final state vecs and add to states.
        airport_vec = o_h_airprt[airprt_idx]
        oh_plane_state = np.concatenate([airport_vec, o_h_plane, noop], axis=0)
        o_h_state.append(oh_plane_state)
        # lookup idxs
        pl_ent = 'P{}'.format(idx)
        planes_dict[pl_ent] = o_h_plane
        # make state dicts and add to states.
        ar_ent = reverse_lookup(airprt_dict, airport_vec)
        idx_state.append('At({}, {})'.format(pl_ent, ar_ent))

    # cargos start end end are mutually exclusive
    # for porpises of this problem. Erp erp.
    for idx, o_h_cargo in enumerate(o_h_cargos):
        # generate random
        init_idx = random.randint(0, num_airport - 1)
        allowed_values = set(range(num_airport))
        #
        allowed_values.discard(init_idx)
        goal_idx = random.choice(list(allowed_values))
        # make one_hot vecs
        air_in_vec = o_h_airprt[init_idx]
        air_gl_vec = o_h_airprt[goal_idx]
        # make final state vecs and add to states.
        oh_init = np.concatenate([air_in_vec, noop, o_h_cargo], axis=0)
        oh_goal = np.concatenate([air_gl_vec, noop, o_h_cargo], axis=0)
        o_h_state.append(oh_init)
        o_h_goals.append(oh_goal)
        # lookup idxs
        cr_ent = 'C{}'.format(idx)
        cargos_dict[cr_ent] = o_h_cargo
        # make state dicts and add to states.
        idx_state.append('At({}, {})'.format(cr_ent, reverse_lookup(airprt_dict, air_in_vec)))
        idx_goals.append('At({}, {})'.format(cr_ent, reverse_lookup(airprt_dict, air_gl_vec)))

    return [[np.asarray(o_h_state), np.asarray(o_h_goals)],
            [idx_state, idx_goals],
            [airprt_dict, cargos_dict, planes_dict]]


def ent_(label, num):
    return '{}{}'.format(label, num)


def entity_ix_generator(label, num_ents, start=0):
    start += 1
    ents_to_ix = {ent_(label, idx): i for idx, i in enumerate(range(start, start + num_ents))}
    ix_to_ents = {i: ent_(label, idx) for idx, i in enumerate(range(start, start + num_ents))}
    return ents_to_ix, ix_to_ents


def entity_2ix_generator(label, type_num, num_ents, start=0):
    start += 1
    ents_to_ix = {ent_(label, idx): (type_num, i) for idx, i in enumerate(range(start, start + num_ents))}
    ix_to_ents = {(type_num, i): ent_(label, idx) for idx, i in enumerate(range(start, start + num_ents))}
    return ents_to_ix, ix_to_ents


def air_cargo_generator_v3(num_airport, num_cargo, num_plane, encoding=1):
    """Init:
        each plane must be at airport
        each cargo must be at an airport
        each cargo has a destination

        Returned typed state expressions with type and instance of each object
        0-0 indicates
          A     P    C
        +---------------+
        | 1-0  2-0  3-0 |
        +---------------+
        """
    if encoding == 1:
        airpt_to_ix, ix_to_airpt = entity_ix_generator("A", num_airport)
        cargo_to_ix, ix_to_cargo = entity_ix_generator("C", num_cargo, start=num_airport)
        plane_to_ix, ix_to_plane = entity_ix_generator("P", num_plane, start=num_cargo + num_airport)
    else:
        airpt_to_ix, ix_to_airpt = entity_2ix_generator("A", 1, num_airport)
        cargo_to_ix, ix_to_cargo = entity_2ix_generator("C", 2, num_cargo)
        plane_to_ix, ix_to_plane = entity_2ix_generator("P", 3, num_plane)

    state_exprs, goal_exprs = [], []
    ents_to_ix = {**airpt_to_ix, **cargo_to_ix, **plane_to_ix}
    ix_to_ents = {**ix_to_airpt, **ix_to_cargo, **ix_to_plane}

    # planes can go wherever
    for idx, plane in ix_to_plane.items():
        # find an airport to put the plane at
        airprt_idx = random.choice(list(airpt_to_ix.keys()))
        state_exprs.append('At({}, {})'.format(plane, airprt_idx))

    # cargos start end end are mutually exclusive
    # for porpises of this problem. Erp erp.
    for idx, cargo in ix_to_cargo.items():
        # generate random
        allowed_values = set(list(airpt_to_ix.keys()))
        init_idx = random.choice(list(allowed_values))
        # set a goal airport
        allowed_values.discard(init_idx)
        goal_idx = random.choice(list(allowed_values))

        # make state dicts and add to states.
        state_exprs.append('At({}, {})'.format(cargo, init_idx))
        goal_exprs.append('At({}, {})'.format(cargo, goal_idx))

    return [[ents_to_ix, ix_to_ents],
            [state_exprs, goal_exprs]]



def arbitrary_ACP(n_airport, n_cargo, n_plane,  one_hot_size=None):
    """
        Generate ACP of arbitrary specified size in a roundabout way.
        Problem Object goes to engine,
        one_hot vecs go to dnc for training.
        """
    # create one hot and entity and state dictionaries
    o_h, k_b, ent_dic = air_cargo_generator_v2(n_airport, n_cargo, n_plane,
                                               one_hot_size=one_hot_size)

    airprt_dict, cargos_dict, planes_dict = ent_dic
    mp_merged = {**airprt_dict, **cargos_dict, **planes_dict}

    airports = list(airprt_dict.keys())
    cargos = list(cargos_dict.keys())
    planes = list(planes_dict.keys())

    init_exprs, goal_exprs = k_b
    o_h_state, o_h_goals = o_h

    pos = [expr(x) for x in init_exprs]
    neg = posneg_helper(cargos, planes, airports, pos)
    init = FluentState(pos, neg)

    goal = [expr(x) for x in goal_exprs]
    acp = AirCargoProblem(cargos, planes, airports, init, goal)
    return (acp, o_h_state, o_h_goals, mp_merged)


def arbitrary_ACP2(n_airport, n_cargo, n_plane, encoding=1):
    """
        Generate ACP of arbitrary specified size in a roundabout way.
        Problem Object goes to engine,
        one_hot vecs go to dnc for training.
        """
    # create one hot and entity and state dictionaries
    (e_ix, ix_e), (state, goals) = air_cargo_generator_v3(n_airport, n_cargo, n_plane, encoding)

    airports = [e for e in e_ix.keys() if 'A' in e]
    cargos = [e for e in e_ix.keys() if 'C' in e]
    planes = [e for e in e_ix.keys() if 'P' in e]

    pos = [expr(x) for x in state]
    neg = posneg_helper(cargos, planes, airports, pos)
    init = FluentState(pos, neg)

    goal = [expr(x) for x in goals]
    acp = AirCargoProblem(cargos, planes, airports, init, goal)
    return acp, (e_ix, ix_e), (pos, goal)
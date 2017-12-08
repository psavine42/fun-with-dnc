import unittest, copy, timeit, time
import dnc_arity_list as dnc
import generators_v2 as gen
import problem.my_air_cargo_problems as mac
from problem.logic import PropKB
from problem.lp_utils import (
    FluentState, encode_state, decode_state,
)
import run, utils
from pprint import  pprint


def test_solve_with_logic(data):
    print('\nWITH LOGICAL HEURISTICS')
    print('GOAL', data.current_problem.goal, '\n')
    start = time.time()
    f = 0
    for n in range(20):
        log_actions = data.best_logic(data.get_raw_actions(mode='all'))
        if log_actions != []:
            print('chosen', log_actions[0])
            data.send_action(data.expr_to_vec(log_actions[0]))
        else:
            f = n
            break
    end = time.time()
    print('DONE in {} steps, {:0.4f} s'.format(f, end - start))


def test_solve_with_algo(data):
    print('\nWITH ASTAR')
    print('GOAL', data.current_problem.goal, '\n')
    f = 0
    start2 = time.time()
    for n in range(20):
        actions = data.get_raw_actions(mode='best')
        # print(actions)
        if actions != []:
            print('chosen', actions[0])
            data.send_action(data.expr_to_vec(actions[0]))
        else:
            f = n
            break
    end2 = time.time()
    print('DONE in {} steps, {:0.4f} s'.format(f, end2 - start2 ))


class Misc(unittest.TestCase):
    def setUp(self):
        self.dataspec = {'solve': True, 'mapping': None,
                         'num_plane': 2, 'one_hot_size': [4, 6, 4, 6, 4, 6],
                         'num_airport': 2, 'plan_phase': 6,
                         'encoding': 2, 'batch_size': 1, 'num_cargo': 2}
        self.dataspec2 = {'solve': True, 'mapping': None,
                          'num_plane': 3, 'one_hot_size': [4, 6, 4, 6, 4, 6],
                          'num_airport': 3, 'plan_phase': 6,
                          'encoding': 2, 'batch_size': 1, 'num_cargo': 3}
    
    def test_cache(self):
        data = gen.AirCargoData(**self.dataspec)
        data.make_new_problem()
        print(data.STATE)
        test_solve_with_logic(copy.deepcopy(data))
        test_solve_with_algo(copy.deepcopy(data))
        # action1 = data.get_raw_actions(mode='best')[0]
        print('\n\n ROUND 2')
        # data2 = gen.AirCargoData(**self.dataspec2)
        data.make_new_problem()
        test_solve_with_logic(copy.deepcopy(data))
        test_solve_with_algo(copy.deepcopy(data))
        # [print(a) for a in best]



    def test_searches(self):
        # utils.get_chkpt('another_test')
        # print('\n---------------------')
        problem = mac.air_cargo_p1()
        ds = decode_state(problem.initial, problem.state_map )





if __name__ == '__main__':
    unittest.main()

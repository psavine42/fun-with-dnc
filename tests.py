import unittest, copy, time
import torch
from problem import generators_v2 as gen
from dnc_arity_list import DNC
import problem.my_air_cargo_problems as mac
from problem.lp_utils import (
    decode_state,
)
import run
from visualize import wuddido as viz


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


sample_args = {'num_plane': 2, 'num_cargo': 2,
               'num_airport': 2,
               'one_hot_size': [4,6, 4, 6, 4, 6],
               'plan_phase': 2 * 3,
               'cuda': False, 'batch_size': 1,
               'encoding': 2, 'solve': True, 'mapping': None}


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
    
    def est_cache(self):
        data = gen.AirCargoData(**self.dataspec)
        data.make_new_problem()
        test_solve_with_logic(copy.deepcopy(data))
        test_solve_with_algo(copy.deepcopy(data))
        print('\n\n ROUND 2')
        data.make_new_problem()
        test_solve_with_logic(copy.deepcopy(data))
        test_solve_with_algo(copy.deepcopy(data))

    def est_searches(self):
        problem = mac.air_cargo_p1()
        ds = decode_state(problem.initial, problem.state_map)


class TestVis(unittest.TestCase):
    def setUp(self):
        self.folder = '1512692566_clip_cont2_40'
        iter = 109 #
        self.base = './models/{}/checkpts/{}/dnc_model.pkl'
        dict1 = torch.load(self.base.format(self.folder, iter))

        self.data = gen.AirCargoData(**sample_args)

        args = run.dnc_args.copy()
        args['output_size'] = self.data.nn_in_size
        args['word_len'] = self.data.nn_out_size

        self.Dnc = DNC(**args)
        self.Dnc.load_state_dict(dict1)
        pass

    def test_show_state(self):
        rand_vec = torch.randn(39, 1)
        viz.ix_to_color(rand_vec)

    def test_run(self):
        record = viz.recorded_step(self.data, self.Dnc)
        viz.make_usage_viz(record)


if __name__ == '__main__':
    unittest.main()















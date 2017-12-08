import unittest
import dnc_arity_list as dnc
import generators_v2 as gen
import problem.my_air_cargo_problems as mac
from problem.lp_utils import (
    FluentState, encode_state, decode_state,
)
import run, utils
from pprint import  pprint


class Misc(unittest.TestCase):
    def setUp(self):
        self.dataspec = {'solve': True, 'mapping': None,
                         'num_plane': 2, 'one_hot_size': [4, 6, 4, 6, 4, 6],
                         'num_airport': 2, 'plan_phase': 6,
                         'encoding': 2, 'batch_size': 1, 'num_cargo': 2}
    
    def test_cache(self):
        datspec = gen.AirCargoData(**self.dataspec)
        print(datspec.STATE)

    def test_searches(self):
        # utils.get_chkpt('another_test')
        problem = mac.air_cargo_p1()
        ds = decode_state(problem.initial, problem.state_map )
        pprint(ds.pos)

if __name__ == '__main__':
    unittest.main()

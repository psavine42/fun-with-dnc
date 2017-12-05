import argparse, os, time

parser = argparse.ArgumentParser(description='Hyperparams')
########### STANDARD ENV SPEC ##############
parser.add_argument('--act', nargs='?', type=str, default='train', help='[]')
parser.add_argument('--load', nargs='?', type=str, default='', help='load model and state')
parser.add_argument('--save', nargs='?', type=str, default='', help='save if true')
parser.add_argument('--iters', nargs='?', type=int, default=21, help='number of iterations')
parser.add_argument('--env', type=str, default='', help='')
parser.add_argument('--lr', type=float, default=1e-5) #1e-5 in paper.
parser.add_argument('--checkpoint_every', type=int, default=1000, help='')
parser.add_argument('--log', nargs='?', type=int, default=0, help='summaries in tb')
parser.add_argument('--notes', nargs='?', type=str, default='', help='any notes')
parser.add_argument('--show_details', nargs='?', type=int, default=1, help='any notes')

########### Algorithm and Optimizer ##############
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--algo', type=str, default='dnc')

########### CONTROL FLOW ##############
parser.add_argument('--feed_last', type=int, default=1, help='')
parser.add_argument('--opt_at', type=str, default='problem', help='')
parser.add_argument('--zero_at', type=str, default='step', help='')
parser.add_argument('--ret_graph', type=int, default=1, help='')
parser.add_argument('--rpkg_step', type=int, default=1, help='')

########### PROBLEM SETUP ##############
parser.add_argument('--n_phases', type=int, default=15)
parser.add_argument('--n_cargo', type=int, default=2)
parser.add_argument('--n_plane', type=int, default=2)
parser.add_argument('--n_airport', type=int, default=2)
parser.add_argument('--typed', nargs='?', type=int, default=1, help='summaries in tb')

########### PROBLEM CONTROL ##############
parser.add_argument('--passing', type=float, default=0.9, help='')
parser.add_argument('--num_tests', type=int, default=2, help='')
parser.add_argument('--num_repeats', type=int, default=2, help='')
parser.add_argument('--max_ents', nargs='?', type=int, default=6, help='summaries in tb')
parser.add_argument('--beta', nargs='?', type=float, default=0.8, help='mixture param from paper')

args = parser.parse_args()

args.repakge_each_step = True if args.rpkg_step == 1 else False
args.ret_graph = True if args.ret_graph == 1 else False

args.prefix = '/output/' if args.env == 'floyd' else './'
if not os.path.exists(args.prefix + 'models'):
    os.mkdir(args.prefix + 'models')

start_timer = time.time()
start = str(start_timer)

if args.save != '':
    args.base_dir = '{}{}{:0.0f}_{}/'.format(args.prefix, 'models/', start, args.save)
    os.mkdir(args.base_dir)
    os.mkdir(args.base_dir + 'checkpts/')
    arg_file = open(args.base_dir + 'params.txt', 'w')
    arg_file.write(str(args))
    arg_file.close()

writer = None
if args.log > 0:
    import logger
    from tensorboardX import SummaryWriter
    logger.log_step += args.log
    writer = SummaryWriter()






#floyd run --env pytorch-0.2 --tensorboard  "bash setup.sh && python run.py --act run --opt_at problem --ret_graph 0 --env floyd --save _nopkg --n_phases 2 --iters 10000"
#python run.py --act run --opt_at problem --ret_graph 0  --save _nopkg --n_phases 2 --iters 10000

# QA training
#floyd run --env pytorch-0.2 --tensorboard "bash setup.sh && python run.py --act dag --iters 1000 --save _new_hidden --ret_graph 1 --opt_at step --env floyd"
#floyd run --cpu --env pytorch-0.2 --tensorboard 'bash setup.sh && python run.py --act dag --iters 1000 --save _new_hidden --ret_graph 1 --opt_at step --zero_at step --env floyd'

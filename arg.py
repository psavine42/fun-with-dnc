import argparse, os, time, json

parser = argparse.ArgumentParser(description='Hyperparams')
########### STANDARD ENV SPEC ##############
parser.add_argument('-a', '--act', nargs='?', type=str, default='train', help='[]')
parser.add_argument('-l', '--load', nargs='?', type=str, default='', help='load model and state')
parser.add_argument('--start_epoch', nargs='?', type=int, default=0, help='load model and state checkpt start')
parser.add_argument('-s', '--save', nargs='?', type=str, default='', help='save if true')
parser.add_argument('-i', '--iters', nargs='?', type=int, default=21, help='number of iterations')
parser.add_argument('--env', type=str, default='', help='')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate 1e-5 in paper')
parser.add_argument('--checkpoint_every', type=int, default=5, help='save models every n epochs')
parser.add_argument('--log', type=int, default=0, help='send to tensorboard frequency. 0 to disable')
parser.add_argument('--notes', type=str, default='', help='any notes')
parser.add_argument('-d', '--show_details', type=int, default=20, help='any notes')

########### Algorithm and Optimizer ##############
parser.add_argument('--opt', type=str, default='adam', help='optimizer to use. options are sgd and adam')
parser.add_argument('--algo', type=str, default='dnc', help='dnc or lstm. default is dnc')
parser.add_argument('-c', '--cuda',  type=int, default=0, help='device to run - 1 if cuda, 0 if cpu')
parser.add_argument('--clip', type=float, default=0, help='gradient clipping, if zero, no clip')

########### CONTROL FLOW ##############
parser.add_argument('--feed_last', type=int, default=1, help='')
parser.add_argument('--opt_at', type=str, default='step', help='')
parser.add_argument('--zero_at', type=str, default='step', help='')
parser.add_argument('--ret_graph', type=int, default=1, help='retain graph todo change to ')
parser.add_argument('--rpkg_step', type=int, default=1, help='repackage input vars at each step')

########### PROBLEM SETUP ##############
parser.add_argument('-p', '--n_phases', type=int, default=15, help='number of training phases')
parser.add_argument('--n_cargo', type=int, default=2, help='number of cargo at starts')
parser.add_argument('--n_plane', type=int, default=2, help='number of plane at starts')
parser.add_argument('--n_airport', type=int, default=2, help='number of airports at starts')
parser.add_argument('-n', '--n_init_start', type=int, default=2, help='number of entities to start with')
parser.add_argument('--typed', type=int, default=1, help='1=use typed entity descriptions, 0=one hot each entity')

########### PROBLEM CONTROL ##############
parser.add_argument('--passing', type=float, default=0.9, help='passing percentage for a run default 0.9')
parser.add_argument('--num_tests', type=int, default=2, help='')
parser.add_argument('--num_repeats', type=int, default=2, help='')
parser.add_argument('--max_ents', type=int, default=6, help='maximum number of entities')
parser.add_argument('--beta', type=float, default=0.8, help='mixture param from paper')

args = parser.parse_args()
print('\n\n')

args.repakge_each_step = True if args.rpkg_step == 1 else False
args.ret_graph = True if args.ret_graph == 1 else False
args.cuda = True if args.cuda == 1 else False
args.clip = None if args.clip == 0 else args.clip
args.prefix = '/output/' if args.env == 'floyd' else './'

if not os.path.exists(args.prefix + 'models'):
    os.mkdir(args.prefix + 'models')

start_timer = time.time()
start = '{:0.0f}'.format(start_timer)

if args.save != '':
    args.base_dir = '{}{}{}_{}/'.format(args.prefix, 'models/', start, args.save)
    os.mkdir(args.base_dir)
    os.mkdir(args.base_dir + 'checkpts/')
    argparse_dict = vars(args)
    with open(args.base_dir + 'params.txt', 'w') as outfile:
        json.dump(argparse_dict, outfile)
    # arg_file = open( 'w')
    # arg_file.write(json.dump(arg_file))
    # arg_file.close()
    print('Saving in folder {}'.format(args.base_dir))

writer = None
if args.log > 0:
    from tensorboardX import SummaryWriter
    global writer
    writer = SummaryWriter()
    from visualize import logger
    logger.log_step += args.log

# test +new no log #
# python run.py -a plan -n 3 -p 2 -d 5 --iters 10 --opt_at step --zero_at step

# test +load #
# python run.py -a plan -n 3 -p 2 -d 5 --load --iters 10 --opt_at step --zero_at step

# test +new +log
# python run.py -a plan -n 3 -p 2 -d 5 --iters 10 --log 2 --opt_at step --zero_at step

# test +new +cuda #
# python run.py -a plan -n 3 -p 2 -d 5 --iters 10 --cuda 1 --opt_at step --zero_at step

# test +new +cuda +clip #
# python run.py -a plan -n 3 -p 2 -d 5 --clip 20 --iters 10 --cuda 1 --opt_at step --zero_at step

# test +new +load +log #
# python run.py -a plan -n 2 --n_phases 2 --show_details 5 --clip 40 --log 2 --iters 10 --cuda 0 --opt_at step --zero_at step




# floyd run --env pytorch-0.2 --tensorboard "bash setup.sh && python run.py --act run --opt_at problem --ret_graph 0 --env floyd --save _nopkg --n_phases 2 --iters 10000"
# python run.py --act run --opt_at problem --ret_graph 0  --save _nopkg --n_phases 2 --iters 10000

# QA training
# floyd run --env pytorch-0.2 --tensorboard "bash setup.sh && python run.py --act dag --iters 1000 --save _new_hidden --ret_graph 1 --opt_at step --env floyd"
# floyd run --cpu --env pytorch-0.2 --tensorboard 'bash setup.sh && python run.py --act dag --iters 1000 --save _new_hidden --ret_graph 1 --opt_at step --zero_at step --env floyd'

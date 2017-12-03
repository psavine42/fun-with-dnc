# fun-with-dnc

Pytorch implementation of deepmind paper "". The code is based on the tensorflow implementation here "". 
Differences:


floyd run --env pytorch-0.2 --tensorboard  "bash setup.sh && python run.py --act run --opt_at problem --ret_graph 0 --env floyd --save _nopkg --n_phases 2 --iters 10000"


python run.py --act run --opt_at problem --ret_graph 0  --save _nopkg --n_phases 2 --iters 10000


QA training
floyd run --env pytorch-0.2 --tensorboard "bash setup.sh && python run.py --act dag --iters 1000 --save _new_hidden --ret_graph 1 --opt_at step --env floyd"

floyd run --cpu --env pytorch-0.2 --tensorboard 'bash setup.sh && python run.py --act dag --iters 1000 --save _new_hidden --ret_graph 1 --opt_at step --zero_at step --env floyd'

#Planning Training

python run.py --act plan --iters 1000  --ret_graph 1 --zero_at step --n_phases 20 --opt_at step

python run.py --act plan --iters 1000  --ret_graph 1 --zero_at step --n_phases 20 --opt_at step --save opt_zero_step
python run.py --act plan --iters 1000  --ret_graph 0 --opt_at problem --save opt_problem_plan --n_phases 20


#testing
#restart best
python run.py --act plan --iters 1000  --ret_graph 0 --opt_at problem --save opt_problem_plan --n_phases 20 --load 1512252569opt_problem_plan_s2_ep_20_gl_158682.pkl


Testing on digit reversing task.
ec02_00-48-05_psavine-G20AJ

Testing on "Air Cargo Prolbem" from 

The Air Cargo problem can be seen as structured prediction.
The DNC can be trained on 


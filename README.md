# fun-with-dnc

Pytorch implementation of deepmind paper "". The code is based on the tensorflow implementation here "". 
Differences:





#Tensorboard
![alt text](images/training.png)



#Planning Training
python run.py --act plan --iters 1000  --ret_graph 1 --zero_at step --n_phases 20 --opt_at step
python run.py --act plan --iters 1000  --ret_graph 1 --zero_at step --n_phases 20 --opt_at step --save opt_zero_step
python run.py --act plan --iters 1000  --ret_graph 0 --opt_at problem --save opt_problem_plan --n_phases 20


#testing
#restart best
python run.py --act plan --iters 1000  --ret_graph 0 --opt_at problem --save opt_problem_plan --n_phases 20 --load the_desc.pkl


#Running on floydhub
Set the --env flag to floyd. When it gets up there, the script will create all the directories in /output. Tensorboard for pytorch does not appear to work on there however.

        floyd run --env pytorch-0.2 --tensorboard "bash setup.sh && python run.py --act dag --iters 1000 --env floyd"


Testing on digit reversing task.
Dec02_00-48-05_psavine-G20AJ

Testing on "Air Cargo Prolbem" from 

The Air Cargo problem can be seen as structured prediction.
The DNC can be trained on 


import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from visualize import logger as sl
import utils as u
import losses as l

def tick(n_total, n_correct, truth, pred):
    n_total += 1
    n_correct += 1 if truth == pred else 0
    return n_total, n_correct

class TrainPlanning:
    def __init__(self, problem, optim, args):
        self.ProblemGen = problem
        self.args = args
        self.optimizer = optim
        self.current_prob = []
        self.reset()

    def reset(self):
        self.num_correct, self.total_moves, self.prev_action = 0, 0, None
        self.n_success = 0

    def phase_fn(self, idx):
        if idx == 0: return self.state_phase
        if idx == 1: return self.goal_phase
        if idx == 2: return self.plan_phase
        if idx == 3: return self.response_phase

    def state_phase(self, Dnc, dnc_state, inputs, mask):
        pass

    def goal_phase(self, Dnc, dnc_state, inputs, mask):
        pass

    def plan_phase(self, Dnc, dnc_state, inputs, mask):
        inputs = Variable(torch.cat([mask_, data.ix_input_to_ixs(prev_action)], 1))
        prev_action, dnc_state = Dnc(inputs, dnc_state)
        pass

    def response_phase(self, Dnc, dnc_state, inputs, mask):
        sl.global_step += 1
        self.total_moves += 1
        final_inputs = Variable(torch.cat([mask, self.ProblemGen.ix_input_to_ixs(prev_action)], 1))

        logits, dnc_state = Dnc(final_inputs, dnc_state)
        expanded_logits = self.ProblemGen.ix_input_to_ixs(logits)
        #
        chosen_act_own, loss_own = L.naive_loss(expanded_logits, all_actions, loss_fn)
        chosen_act_star, loss_star =  L.naive_loss(expanded_logits, targets_star, loss_fn, log_itr=sl.global_step)

        # set next input to be the networks current action ...
        if random.random() < self.args.beta:
            loss = loss_star
            final_action = chosen_act_star
        else:
            loss = loss_own
            final_action = chosen_act_own
        self.num_correct += 1 if chosen_act_own == final_action else 0
        return Dnc, dnc_state, final_action

    def train(self, Dnc):
        for n in range(self.args.iters):
            self.current_prob = self.ProblemGen.make_new_problem()
            dnc_state = Dnc.init_state(grad=False)
            self.optimizer.zero_grad()

            for idx, _ in enumerate(self.current_prob):
                inputs, mask = self.ProblemGen.getitem()
                phase_fn = self.phase_fn(idx)
                phase_fn(Dnc, dnc_state, inputs, mask)
        pass

    def step(self):
        pass

    def end_problem(self):
        print("solved {} out of {} -> {}".format(self.n_success, self.args.iters, self.n_success / self.args.iters))
        pass



def play_qa_readable(args, data, DNC):
    criterion = nn.CrossEntropyLoss()
    cum_correct, cum_total = [], []

    for trial in range(args.iters):
        phase_masks = data.make_new_problem()
        n_total, n_correct, loss = 0, 0, 0
        dnc_state = DNC.init_state(grad=False)


        for phase_idx in phase_masks:
            if phase_idx == 0 or phase_idx == 1:

                inputs, msk = data.getitem()
                print(data.human_readable(inputs, msk))

                inputs = Variable(torch.cat([msk, inputs], 1))
                logits, dnc_state = DNC(inputs, dnc_state)
            else:
                final_moves = data.get_actions(mode='one')
                if final_moves == []:
                    break
                data.send_action(final_moves[0])
                mask = data.phase_oh[2].unsqueeze(0)
                vec = data.vec_to_ix(final_moves[0])
                print('\n')
                print(data.human_readable(vec, mask))

                inputs2 = Variable(torch.cat([mask, vec], 1))
                logits, dnc_state = DNC(inputs2, dnc_state)

                for _ in range(args.num_tests):
                    # ask where is ---?

                    masked_input, mask_chunk, ground_truth = data.masked_input()
                    print("Context:", data.human_readable(ground_truth))
                    print("Q:")

                    logits, dnc_state = DNC(Variable(masked_input), dnc_state)
                    expanded_logits = data.ix_input_to_ixs(logits)

                    #losses
                    lstep = l.action_loss(expanded_logits, ground_truth, criterion, log=True)

                    #update counters
                    prediction = u.get_prediction(expanded_logits, [3, 4])
                    print("A:")
                    n_total, n_correct = tick(n_total, n_correct, mask_chunk, prediction)
                    print("correct:", mask_chunk == prediction)


        cum_total.append(n_total)
        cum_correct.append(n_correct)
        sl.writer.add_scalar('recall.pct_correct', n_correct / n_total, sl.global_step)
        print("trial: {}, step:{}, accy {:0.4f}, cum_score {:0.4f}, loss: {:0.4f}".format(
            trial, sl.global_step, n_correct / n_total, u.running_avg(cum_correct, cum_total), loss.data[0]))
    return DNC,  dnc_state, u.running_avg(cum_correct, cum_total)


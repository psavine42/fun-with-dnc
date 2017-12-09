import dnc_arity_list as dnc
import numpy as np
from utils import running_avg, flat, save, _variable
import utils as u
import torch
import torch.nn as nn
from problem import generators_v2 as gen
import torch.optim as optim
import time, random
from visualize import logger as sl
import os
import losses as L
from arg import args

random.seed()
batch_size = 1
dnc_args = {'num_layers': 2,
            'num_read_heads': 2,
            'hidden_size': 250,
            'num_write_heads': 1,
            'memory_size':  100, #50
            'batch_size': batch_size}


def generate_data_spec(args, num_ents=2, solve=True):
    if args.typed is 1:
        ix_size = [4, args.max_ents, 4, args.max_ents, 4, args.max_ents]
        encoding = 2
    else:
        ix_state = args.max_ents * 3
        ix_size = [ix_state, ix_state, ix_state]
        encoding = 1
    return {'num_plane': num_ents, 'num_cargo': num_ents, 'num_airport': num_ents,
            'one_hot_size': ix_size, 'plan_phase': num_ents * 3, 'cuda': args.cuda,
            'batch_size': 1, 'encoding': encoding, 'solve': solve, 'mapping': None}


def setupLSTM(args):
    data = gen.AirCargoData(**generate_data_spec(args))
    dnc_args['output_size'] = data.nn_in_size  # output has no phase component
    dnc_args['word_len'] = data.nn_out_size
    print(dnc_args)
    # input_size = self.output_size + word_len * num_read_heads
    Dnc = dnc.VanillaLSTM(batch_size=1, num_layers=2, input_size=data.nn_in_size,
                          output_size=data.nn_out_size, hidden_size=250, num_reads=2)
    previous_out, (ho1, hc1), (ho2, hc2) = Dnc.init_state()
    if args.opt == 'adam':
        optimizer = optim.Adam([{'params': Dnc.parameters()}, {'params': ho1},
                                {'params': hc1}, {'params': hc2}], lr=args.lr)
    else:
        optimizer = optim.SGD([{'params': Dnc.parameters()}, {'params': ho1},
                               {'params': hc1}, {'params': hc2}], lr=args.lr)

    lstm_state = (previous_out, (ho1, hc1), (ho2, hc2))
    return data, Dnc, optimizer, lstm_state


def setupDNC(args):
    """
        Loader for files or setup new DNC and optimizer
    :param args:
    :return:
    """
    if args.algo == 'lstm':
        return setupLSTM(args)
    data = gen.AirCargoData(**generate_data_spec(args))
    dnc_args['output_size'] = data.nn_in_size  # output has no phase component
    dnc_args['word_len'] = data.nn_out_size
    print('dnc_args:\n', dnc_args, '\n')
    if args.load == '':
        Dnc = dnc.DNC(**dnc_args)
        if args.opt == 'adam':
            optimizer = optim.Adam(Dnc.parameters(), lr=args.lr)
        elif args.opt == 'sgd':
            optimizer = optim.SGD(Dnc.parameters(), lr=args.lr)
        else:
            optimizer = None
    else:
        model_path, optim_path = u.get_chkpt(args.load)
        print('loading', model_path)
        Dnc = dnc.DNC(**dnc_args)
        Dnc.load_state_dict(torch.load(model_path))

        optimizer = optim.Adam(Dnc.parameters(), lr=args.lr)
        if os.path.exists(optim_path):
            optimizer.load_state_dict(torch.load(optim_path))

    if args.cuda is True:
        Dnc = Dnc.cuda()
    lstm_state = Dnc.init_rnn()
    return data, Dnc, optimizer, lstm_state


def tick(n_total, n_correct, truth, pred):
    n_total += 1
    n_correct += 1 if truth == pred else 0
    sl.global_step += 1
    return n_total, n_correct


def train_qa2(args, data, DNC, optimizer):
    """
        I am jacks liver. This is a sanity test

        0 - describe state.
        1 - describe goal.
        2 - do actions.
        3 - ask some questions
        :param args:
        :return:
        """
    criterion = nn.CrossEntropyLoss()
    cum_correct, cum_total = [], []

    for trial in range(args.iters):
        phase_masks = data.make_new_problem()
        n_total, n_correct, loss = 0, 0, 0
        dnc_state = DNC.init_state(grad=False)
        optimizer.zero_grad()

        for phase_idx in phase_masks:
            if phase_idx == 0 or phase_idx == 1:
                inputs = _variable(data.getitem_combined())
                logits, dnc_state = DNC(inputs, dnc_state)
            else:
                final_moves = data.get_actions(mode='one')
                if final_moves == []:
                    break
                data.send_action(final_moves[0])
                mask = data.phase_oh[2].unsqueeze(0)
                inputs2 = _variable(torch.cat([mask, data.vec_to_ix(final_moves[0])], 1))
                logits, dnc_state = DNC(inputs2, dnc_state)

                for _ in range(args.num_tests):
                    # ask where is ---?
                    if args.zero_at == 'step':
                        optimizer.zero_grad()
                    masked_input, mask_chunk, ground_truth = data.masked_input()
                    logits, dnc_state = DNC(_variable(masked_input), dnc_state)
                    expanded_logits = data.ix_input_to_ixs(logits)

                    # losses
                    lstep = L.action_loss(expanded_logits, ground_truth, criterion, log=True)
                    if args.opt_at == 'problem':
                        loss += lstep
                    else:
                        lstep.backward(retain_graph=args.ret_graph)
                        optimizer.step()
                        loss = lstep

                    # update counters
                    prediction = u.get_prediction(expanded_logits, [3, 4])
                    n_total, n_correct = tick(n_total, n_correct, mask_chunk, prediction)

        if args.opt_at == 'problem':
            loss.backward(retain_graph=args.ret_graph)
            optimizer.step()
            sl.writer.add_scalar('losses.end', loss.data[0], sl.global_step)

        cum_total.append(n_total)
        cum_correct.append(n_correct)
        sl.writer.add_scalar('recall.pct_correct', n_correct / n_total, sl.global_step)
        print("trial: {}, step:{}, accy {:0.4f}, cum_score {:0.4f}, loss: {:0.4f}".format(
            trial, sl.global_step, n_correct / n_total, running_avg(cum_correct, cum_total), loss.data[0]))
    return DNC, optimizer, dnc_state, running_avg(cum_correct, cum_total)


def random_seq(args, data, DNC, lstm_state, optimizer):
    pass


def train_rl(args, data, DNC, lstm_state, optimizer):
    """

    :param args:
    :param data:
    :param DNC: a tuple of value and action networks
    :param lstm_state:
    :param optimizer:
    :return:
    """
    for trial in range(args.iters):
        start_prob = time.time()
        phase_masks = data.make_new_problem()


    pass


def train_plan(args, data, DNC, lstm_state, optimizer):
    """
        Things to test after some iterations:
         - on planning phase and on

         with goals - chose a goal and work toward that
        :param args:
        :return:
        """
    criterion = nn.CrossEntropyLoss().cuda() if args.cuda is True else nn.CrossEntropyLoss()
    cum_correct, cum_total, prob_times, n_success = [], [], [], 0
    penalty = 1.1

    for trial in range(args.iters):
        start_prob = time.time()
        phase_masks = data.make_new_problem()
        n_total, n_correct, prev_action, loss, stats = 0, 0, None, 0, []
        dnc_state = DNC.init_state(grad=False)
        lstm_state = DNC.init_rnn(grad=False) # lstm_state, 
        optimizer.zero_grad()

        for phase_idx in phase_masks:

            if phase_idx == 0 or phase_idx == 1:
                inputs = _variable(data.getitem_combined())
                logits, dnc_state, lstm_state = DNC(inputs, lstm_state, dnc_state)
                _, prev_action = data.strip_ix_mask(logits)

            elif phase_idx == 2:
                mask = _variable(data.getmask())
                inputs = torch.cat([mask, prev_action], 1)
                logits, dnc_state, lstm_state = DNC(inputs, lstm_state, dnc_state)
                _, prev_action = data.strip_ix_mask(logits)

            else:
                # sample from best moves
                actions_star, all_actions = data.get_actions(mode='both')
                if not actions_star:
                    break
                if args.zero_at == 'step':
                    optimizer.zero_grad()

                mask = data.getmask()
                prev_action = prev_action.cuda() if args.cuda is True else prev_action
                pr = u.depackage(prev_action)

                final_inputs = _variable(torch.cat([mask, pr], 1))
                logits, dnc_state, lstm_state = DNC(final_inputs, lstm_state, dnc_state)
                exp_logits = data.ix_input_to_ixs(logits)

                guided = random.random() < args.beta
                # thing 1
                if guided: # guided loss
                    final_action, lstep = L.naive_loss(exp_logits, actions_star, criterion, log=True)
                else: # pick own move
                    final_action, lstep = L.naive_loss(exp_logits, all_actions, criterion, log=True)

                # penalty for todo tests this !!!!
                action_own = u.get_prediction(exp_logits)
                if args.penalty and not [tuple(flat(t)) for t in all_actions]:
                    final_loss = lstep * _variable([args.penalty])
                else:
                    final_loss = lstep

                if args.opt_at == 'problem':
                    loss += final_loss
                else:

                    final_loss.backward(retain_graph=args.ret_graph)
                    if args.clip:
                        torch.nn.utils.clip_grad_norm(DNC.parameters(), args.clip)
                    optimizer.step()
                    loss = lstep

                data.send_action(final_action)

                if (trial + 1) % args.show_details == 0:
                    action_accs = u.human_readable_res(data, all_actions, actions_star,
                                                       action_own, guided, lstep.data[0])
                    stats.append(action_accs)
                n_total, _ = tick(n_total, n_correct, action_own, flat(final_action))
                n_correct += 1 if action_own in [tuple(flat(t)) for t in actions_star] else 0
                prev_action = data.vec_to_ix(final_action)

        if stats:
            arr = np.array(stats)
            correct = len([1 for i in list(arr.sum(axis=1)) if i == len(stats[0])]) / len(stats)
            sl.log_acc(list(arr.mean(axis=0)), correct)

        if args.opt_at == 'problem':
            floss = loss / n_total
            floss.backward(retain_graph=args.ret_graph)
            if args.clip:
                torch.nn.utils.clip_grad_norm(DNC.parameters(), args.clip)
            optimizer.step()
            sl.writer.add_scalar('losses.end', floss.data[0], sl.global_step)

        n_success += 1 if n_correct / n_total > args.passing else 0
        cum_total.append(n_total)
        cum_correct.append(n_correct)
        sl.add_scalar('recall.pct_correct', n_correct / n_total, sl.global_step)
        print("trial {}, step {} trial accy: {}/{}, {:0.2f}, running total {}/{}, running avg {:0.4f}, loss {:0.4f}  ".format(
            trial, sl.global_step, n_correct, n_total, n_correct / n_total, n_success, trial,
            running_avg(cum_correct, cum_total), loss.data[0]
            ))
        end_prob = time.time()
        prob_times.append(start_prob - end_prob)
    print("solved {} out of {} -> {}".format(n_success, args.iters, n_success / args.iters))
    return DNC, optimizer, lstm_state, running_avg(cum_correct, cum_total)


def train_manager(args, train_fn):
    """

    :param args: args object. see arg.py or run.py -h for details
    :param train_fn: the training function -
    :return:
    """
    datspec = generate_data_spec(args)
    print('\nInitial Spec', datspec)

    _, DNC, optimizer, lstm_state = setupDNC(args)
    start_ents, score, global_epoch = args.n_init_start, 0, args.start_epoch
    print('\nDnc structure', DNC)

    for problem_size in range(args.max_ents):
        test_size = problem_size + start_ents
        passing = False
        data_spec = generate_data_spec(args, num_ents=test_size, solve=test_size * 3)
        data = gen.AirCargoData(**data_spec)

        print("beginning new training Size: {}".format(test_size))
        for train_epoch in range(args.n_phases):
            ep_start = time.time()
            global_epoch += 1
            print("\nStarting Epoch {}".format(train_epoch))

            DNC, optimizer, lstm_state, score = train_fn(args, data, DNC, lstm_state, optimizer)
            if (train_epoch + 1) % args.checkpoint_every and args.save != '':
                save(DNC, optimizer, lstm_state, args, global_epoch)

            ep_end = time.time()
            ttl_s = ep_end - ep_start
            print('finished epoch: {}, score: {}, ttl-time: {:0.4f}, time/prob: {:0.4f}'.format(
                train_epoch, score, ttl_s, ttl_s / args.iters
            ))
            if score > args.passing:
                print('model_successful: {}, {} '.format(score, train_epoch))
                print('----------------------WOO!!--------------------------')
                passing = True
                break

        if passing is False:
            print("Training has FAILED for problem of size: {}, after {} epochs of {} phases".format(
                test_size, args.max_ents, args.n_phases
            ))
            print("final score was {}".format(score))
            break


if __name__== "__main__":
    print(args)
    if args.act == 'plan':
        train_manager(args, train_plan)
    elif args.act == 'qa':
        train_manager(args, train_qa2)
    elif args.act == 'clean':
        pass
    else:
        print("wrong action")
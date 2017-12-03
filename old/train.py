
def train_qa(args, num_problems, data, Dnc, optimizer, dnc_state, save_=False):
    """
    I am jacks liver. This is a sanity test

    0 - describe state.
    1 - describe goal.
    2 - do actions.
    3 - ask some questions
    :param args:
    :return:
    """
    sl.log_step += args.log

    print(Dnc)
    criterion = nn.CrossEntropyLoss()

    cum_correct = []
    cum_total_move = []
    num_tests = 2
    num_repeats = 1
    for n in range(num_problems):
        masks = data.make_new_problem()
        num_correct = 0
        total_moves = 0
        # prev_action = None

        if args.repakge_each_step is False:
            dnc_state = dnc.repackage(dnc_state)
            optimizer.zero_grad()
        # repackage the state to zero grad at start of each problem

        for idx, mask in enumerate(masks):
            sl.global_step += 1
            inputs, mask_ = data.getitem()

            if mask == 0 or mask == 1:
                for nz in range(num_repeats):
                    inputs1 = Variable(torch.cat([mask_, inputs], 1))
                    logits, dnc_state = Dnc(inputs1, dnc_state)

                # sl.log_state(dnc_state)
            else:
                targets_star = data.get_actions(mode='one')
                if targets_star == []:
                    break
                final_move = targets_star[0]

                if args.repakge_each_step is True:
                    dnc_state = dnc.repackage(dnc_state)
                    optimizer.zero_grad()

                for nq in range(num_repeats):
                    inputs2 = Variable(torch.cat([data.phase_oh[2].unsqueeze(0), data.vec_to_ix(final_move)], 1))
                    logits, dnc_state = Dnc(inputs2, dnc_state)

                data.send_action(final_move)
                # sl.log_state(dnc_state)

                for _ in range(num_tests):
                    total_moves += 1
                    state_expr = random.choice(data.pull_state())
                    state_vec = data.expr_to_vec(state_expr)
                    mask_idx = 2  # random.randint(1, 2)
                    mask_chunk = state_vec[mask_idx]

                    zeros = 0 if type(mask_chunk) == int else tuple([0] * len(mask_chunk))
                    masked_state_vec = state_vec.copy()
                    # masked_state_vec[mask_idx] = zeros
                    masked_state_vec[2] = zeros

                    inputs3 = Variable(torch.cat([data.phase_oh[3].unsqueeze(0), data.vec_to_ix(masked_state_vec)], 1))
                    logits, dnc_state = Dnc(inputs3, dnc_state)

                    expanded_logits = data.ix_input_to_ixs(logits)
                    idx1, idx2 = mask_idx * 2 - 1, mask_idx * 2

                    target_chunk1 = Variable(torch.LongTensor([mask_chunk[0]]))
                    target_chunk2 = Variable(torch.LongTensor([mask_chunk[1]]))

                    loss1 = criterion(expanded_logits[idx1], target_chunk1)
                    loss2 = criterion(expanded_logits[idx2], target_chunk2)
                    loss = loss1 + loss2
                    sl.log_state(dnc_state)

                    loss.backward(retain_graph=args.ret_graph)
                    optimizer.step()
                    sl.log_state(dnc_state)

                    resp1, pidx1 = expanded_logits[idx1].data.topk(1)
                    resp2, pidx2 = expanded_logits[idx2].data.topk(1)
                    pred_tuple = pidx1.squeeze()[0], pidx2.squeeze()[0]
                    # pred_expr = data.lookup_ix_to_expr(pred_tuple)
                    correct_ = mask_chunk == pred_tuple
                    num_correct += 1 if mask_chunk == pred_tuple else 0
                    # print("step {}.{}, loss: {:0.2f}, state {} actual {} pred: {}, {}".format(
                    # n, idx, loss.data[0], state_expr, mask_chunk, pred_tuple, correct_))
                    sl.log_loss_qa(loss1, loss2, loss)

        cum_total_move.append(total_moves)
        cum_correct.append(num_correct)
        trial_acc = num_correct / total_moves
        sl.writer.add_scalar('recall.pct_correct', trial_acc, sl.global_step)
        sl.log_state(dnc_state)
        sl.log_model(Dnc)

        print("trial: {} accy {:0.4f}, cum_score {:0.4f}".format(n, trial_acc, sum(cum_correct[-100:]) / sum(cum_total_move[-100:])))
    if save_ is not False:
        save(Dnc, dnc_state, start, args.save, sl.global_step)

    score = sum(cum_correct[-100:]) / sum(cum_total_move[-100:])
    return Dnc, optimizer, dnc_state, score

def setupDNC(args):
    if args.algo == 'lstm':
        return setupLSTM(args)
    data = gen.AirCargoData(**generate_data_spec(args))
    dnc_args['output_size'] = data.nn_in_size  # output has no phase component
    dnc_args['word_len'] = data.nn_out_size
    print(dnc_args)
    if args.load == '':
        Dnc = dnc.DNC(**dnc_args)
        o, m, r, w, l, lw, u, (ho, hc) = Dnc.init_state()
    else:
        model_path = './models/dnc_model_' + args.load
        state_path = './models/dnc_state_' + args.load
        print('loading', model_path, state_path)
        Dnc = dnc.DNC(**dnc_args)
        Dnc.load_state_dict(torch.load(model_path))
        o, m, r, w, l, lw, u, (ho, hc) = torch.load(state_path)
        print(dnc_checksum([o, m, r, w, l, lw, u]))

    lr = 5e-5 if args.lr is None else args.lr
    if args.opt == 'adam':
        optimizer = optim.Adam([{'params': Dnc.parameters()}, {'params': o}, {'params': m}, {'params': r}, {'params': w},
                               {'params': l}, {'params': lw}, {'params': u}, {'params': ho}, {'params': hc}],
                               lr=lr)
    else:
        optimizer = optim.SGD([{'params': Dnc.parameters()}, {'params': o}, {'params': m}, {'params': r}, {'params': w},
                               {'params': l}, {'params': lw}, {'params': u}, {'params': ho}, {'params': hc}],
                               lr=lr)
    dnc_state = (o, m, r, w, l, lw, u, (ho, hc))
    return data, Dnc, optimizer, dnc_state


"""
                    #target_chunk1 = Variable(torch.LongTensor([mask_chunk[0]]))
                    #target_chunk2 = Variable(torch.LongTensor([mask_chunk[1]]))
                    #loss1 = criterion(expanded_logits[idx1], target_chunk1)
                    #loss2 = criterion(expanded_logits[idx2], target_chunk2)
                    #lstep = loss1 + loss2
                    # action_loss(logits, expanded_logits, criterion)
"""
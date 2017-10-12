
"interface"
def _read_inputs(self, inputs):
    """Applies transformations to `inputs` to get control for this module."""

    def _linear(first_dim, second_dim, name, activation=None):
      """Returns a linear transformation of `inputs`, followed by a reshape."""
      linear = snt.Linear(first_dim * second_dim, name=name)(inputs)
      if activation is not None:
        linear = activation(linear, name=name + '_activation')
      return tf.reshape(linear, [-1, first_dim, second_dim])

    # v_t^i - The vectors to write to memory, for each write head `i`.
    write_vectors = _linear(self._num_writes, self._word_size, 'write_vectors')

    # e_t^i - Amount to erase the memory by before writing, for each write head.
    erase_vectors = _linear(self._num_writes, self._word_size, 'erase_vectors',
                            tf.sigmoid)

    # f_t^j - Amount that the memory at the locations read from at the previous
    # time step can be declared unused, for each read head `j`.
    free_gate = tf.sigmoid(
        snt.Linear(self._num_reads, name='free_gate')(inputs))

    # g_t^{a, i} - Interpolation between writing to unallocated memory and
    # content-based lookup, for each write head `i`. Note: `a` is simply used to
    # identify this gate with allocation vs writing (as defined below).
    allocation_gate = tf.sigmoid(
        snt.Linear(self._num_writes, name='allocation_gate')(inputs))

    # g_t^{w, i} - Overall gating of write amount for each write head.
    write_gate = tf.sigmoid(
        snt.Linear(self._num_writes, name='write_gate')(inputs))

    # \pi_t^j - Mixing between "backwards" and "forwards" positions (for
    # each write head), and content-based lookup, for each read head.
    num_read_modes = 1 + 2 * self._num_writes
    read_mode = snt.BatchApply(tf.nn.softmax)(
        _linear(self._num_reads, num_read_modes, name='read_mode'))

    # Parameters for the (read / write) "weights by content matching" modules.
    write_keys = _linear(self._num_writes, self._word_size, 'write_keys')
    write_strengths = snt.Linear(self._num_writes, name='write_strengths')(
        inputs)

    #k_t^
    read_keys = _linear(self._num_reads, self._word_size, 'read_keys')
    read_strengths = snt.Linear(self._num_reads, name='read_strengths')(inputs)

    result = {
        'read_content_keys': read_keys,
        'read_content_strengths': read_strengths,
        'write_content_keys': write_keys,
        'write_content_strengths': write_strengths,
        'write_vectors': write_vectors,
        'erase_vectors': erase_vectors,
        'free_gate': free_gate,
        'allocation_gate': allocation_gate,
        'write_gate': write_gate,
        'read_mode': read_mode,
    }
    return result



    def _build(self, inputs, prev_state):
    """Connects the DNC core into the graph.

      Args:
        inputs: Tensor input.
        prev_state: A `DNCState` tuple containing the fields `access_output`,
            `access_state` and `controller_state`. `access_state` is a 3-D Tensor
            of shape `[batch_size, num_reads, word_size]` containing read words.
            `access_state` is a tuple of the access module's state, and
            `controller_state` is a tuple of controller module's state.

      Returns:
        A tuple `(output, next_state)` where `output` is a tensor and `next_state`
        is a `DNCState` tuple containing the fields `access_output`,
        `access_state`, and `controller_state`.
      """

    prev_access_output = prev_state.access_output
    prev_access_state = prev_state.access_state
    prev_controller_state = prev_state.controller_state

    batch_flatten = snt.BatchFlatten()
    controller_input = tf.concat(
        [batch_flatten(inputs), batch_flatten(prev_access_output)], 1)

    controller_output, controller_state = self._controller(
        controller_input, prev_controller_state)

    controller_output = self._clip_if_enabled(controller_output)
    controller_state = snt.nest.map(self._clip_if_enabled, controller_state)

    access_output, access_state = self._access(controller_output,
                                               prev_access_state)

    output = tf.concat([controller_output, batch_flatten(access_output)], 1)
    output = snt.Linear(
        output_size=self._output_size.as_list()[0],
        name='output_linear')(output)
    output = self._clip_if_enabled(output)

    return output, DNCState(
        access_output=access_output,
        access_state=access_state,
        controller_state=controller_state)
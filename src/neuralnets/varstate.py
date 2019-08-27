import tensorflow as tf

class VariableState:
    """
    Manage the state of a set of variables.
    """
    def __init__(self, session, variables):
        self._session = session
        self._variables = variables
        self._placeholders = [tf.compat.v1.placeholder(v.dtype.base_dtype, shape=v.get_shape())
                              for v in variables]
        assigns = [tf.compat.v1.assign(v, p) for v, p in zip(self._variables, self._placeholders)]
        self._assign_op = tf.group(*assigns)

    def export_variables(self):
        """
        Save the current variables.
        """
        return self._session.run(self._variables)

    def import_variables(self, values):
        """
        Restore the variables.
        """
        self._session.run(self._assign_op, feed_dict=dict(zip(self._placeholders, values)))

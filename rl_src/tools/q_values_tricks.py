import numpy as np


class QValuesTricks():
    @staticmethod
    def make_qvalues_tensorable(qvalues_table):
        nr_states = len(qvalues_table)
        for state in range(nr_states):
            memory_size = len(qvalues_table[state])
            for memory in range(memory_size):
                if qvalues_table[state][memory] == None:
                    not_none_values = [qvalues_table[state][i] for i in range(
                        memory_size) if qvalues_table[state][i] is not None]
                    if len(not_none_values) == 0:
                        qvalues_table[state][memory] = 0.0
                    else:
                        qvalues_table[state][memory] = np.min(not_none_values)
        return qvalues_table

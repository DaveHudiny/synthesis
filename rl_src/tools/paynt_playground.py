import paynt
import os


class PAYNT_Playground:
    @staticmethod
    def fill_nones_in_qvalues(qvalues):
        for state in range(len(qvalues)):
            for memory in range(len(qvalues[state])):
                if qvalues[state][memory] is None:
                    qvalues[state][memory] = np.mean([qvalues[state][i] for i in range(
                        len(qvalues[state])) if qvalues[state][i] is not None])
        return qvalues

    # Not a good implementation, if we work in a loop with multiple different models.
    @classmethod
    def singleton_init_models(cls, sketch_path, properties_path):
        if not os.path.exists(sketch_path):
            raise ValueError(f"Sketch file {sketch_path} does not exist.")
        if not hasattr(cls, "quotient") and not hasattr(cls, "synthesizer"):
            cls.quotient = paynt.parser.sketch.Sketch.load_sketch(
                sketch_path, properties_path)
            cls.k = 3  # May be unknown?
            cls.quotient.set_imperfect_memory_size(cls.k)
            cls.synthesizer = paynt.synthesizer.synthesizer_pomdp.SynthesizerPomdp(
                cls.quotient, method="ar", storm_control=None)

    @classmethod
    def compute_qvalues_function(cls):
        assignment = cls.synthesizer.synthesize()
        # before the quotient is modified we can use this assignment to compute Q-values
        assert assignment is not None, "Provided assignment cannot be None."
        qvalues = cls.quotient.compute_qvalues(assignment)
        # note Q(s,n) may be None if (s,n) exists in the unfolded POMDP but is not reachable in the induced DTMC
        memory_size = len(qvalues[0])
        assert cls.k == memory_size
        qvalues = PAYNT_Playground.fill_nones_in_qvalues(qvalues)
        return qvalues

    @classmethod
    def get_fsc_critic_components(cls, sketch_path, properties_path):
        cls.singleton_init_models(
            sketch_path=sketch_path, properties_path=properties_path)
        qvalues = cls.compute_qvalues_function()
        action_labels_at_observation = cls.quotient.action_labels_at_observation
        return qvalues, action_labels_at_observation

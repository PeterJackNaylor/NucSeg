import tensorflow as tf
from tensorflow.keras.metrics import Metric, Precision, Recall, Accuracy
from tensorflow.python.keras import backend
from tensorflow.python.ops import math_ops
class AccuracyDistance(Metric):

    def __init__(self, name='accuracy_d', **kwargs):
        super(AccuracyDistance, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name='acc_d', initializer='zeros')
        self.accuracy_fn = Accuracy()
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)
        acc = self.accuracy_fn(y_true, y_pred)
        self.accuracy.assign(tf.subtract(1., acc))

    def result(self):
        return self.accuracy

    def reset_states(self):
        # we also need to reset the state of accuracy objects
        self.accuracy_fn.reset_states()
        self.accuracy.assign(0)

class F1_ScoreDistance(Metric):

    def __init__(self, name='f1_score_d', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1_d', initializer='zeros')
        self.precision_fn = Precision(thresholds=0.5)
        self.recall_fn = Recall(thresholds=0.5)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        p = self.precision_fn(y_true, y_pred)
        r = self.recall_fn(y_true, y_pred)
        # since f1 is a variable, we use assign
        self.f1.assign(2 * ((p * r) / (p + r + 1e-6)))

    def result(self):
        return self.f1

    def reset_states(self):
        # we also need to reset the state of the precision and recall objects
        self.precision_fn.reset_states()
        self.recall_fn.reset_states()
        self.f1.assign(0)

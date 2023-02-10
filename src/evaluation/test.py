import numpy as np
from evaluation_mit1003 import EvaluationMetric


metrics = EvaluationMetric()
a = metrics.string_based_time_delay_embedding_distance(np.array([1,0,6,6]),np.array([1,0,1,1]))
print(a)

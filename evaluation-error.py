import numpy as np
import pandas as pd
import implicit
from implicit import evaluation as ev
import time
import scipy
import random
from sklearn import metrics
import scipy.sparse as sparse

df_wide = pd.read_csv("https://raw.githubusercontent.com/murphystout/implicit-feedback-ml/main/df_wide.csv")
csr = scipy.sparse.csr_matrix(df_wide.values)
train, test = ev.train_test_split(csr)
model = implicit.als.AlternatingLeastSquares()

model.fit(train)
ev.ranking_metrics_at_k(model = model, train_user_items = train, test_user_items = test)

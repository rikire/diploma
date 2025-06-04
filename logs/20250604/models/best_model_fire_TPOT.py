import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import ElasticNetCV, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MaxAbsScaler
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: -2460.677658329298
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=SGDRegressor(alpha=0.0, eta0=0.01, fit_intercept=False, l1_ratio=0.5, learning_rate="invscaling", loss="epsilon_insensitive", penalty="elasticnet", power_t=10.0)),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.8500000000000001, tol=0.01)),
    MaxAbsScaler(),
    Nystroem(gamma=0.9, kernel="rbf", n_components=4),
    SelectFwe(score_func=f_regression, alpha=0.021),
    GradientBoostingRegressor(alpha=0.95, learning_rate=1.0, loss="huber", max_depth=3, max_features=0.5, min_samples_leaf=6, min_samples_split=6, n_estimators=100, subsample=0.9000000000000001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

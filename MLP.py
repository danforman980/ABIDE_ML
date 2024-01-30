import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, recall_score, accuracy_score, precision_score

scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score)
}

df= pd.read_csv('correlation_matrix_data_flattened_PCA.csv', index_col = 0)

x = df.drop('diagnosis',axis=1)

y = df['diagnosis']

params = {"hidden_layer_sizes": [(50,)], "activation": ["logistic"], 'solver': ['lbfgs',], 'alpha': [0.0005], 'max_iter': [2000], 'learning_rate': ['adaptive']}
skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 10)
dl = MLPClassifier()
dl_opt = GridSearchCV(dl,params ,refit='accuracy_score',verbose=3, scoring=scorers, cv = skf)
dl_opt.fit(x, y)

cv_results = pd.DataFrame.from_dict(dl_opt.cv_results_)
df2 = cv_results[['mean_test_precision_score', 'mean_test_accuracy_score', 'mean_test_recall_score']]
maxacc = cv_results[['split0_test_precision_score', 'split1_test_precision_score', 'split2_test_precision_score', 'split3_test_precision_score', 'split4_test_precision_score', 'split5_test_precision_score', 'split6_test_precision_score', 'split7_test_precision_score', 'split8_test_precision_score', 'split9_test_precision_score']]



pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('expand_frame_repr', False)

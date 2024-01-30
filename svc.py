import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, recall_score, accuracy_score, precision_score

scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score)
}

df= pd.read_csv('correlation_matrix_data_flattened_PCA.csv', index_col = 0)

x = df.drop('diagnosis',axis=1)

y = df['diagnosis']

params = {'C': [10], 'gamma': ['scale'],'kernel':  ['sigmoid',]}
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 10)
svc_opt = GridSearchCV(SVC(),params,refit='accuracy_score',verbose=3, scoring=scorers, cv = skf)
svc_opt.fit(x,y)

cv_results = pd.DataFrame.from_dict(svc_opt.cv_results_)
df2 = cv_results[['mean_test_precision_score', 'mean_test_accuracy_score', 'mean_test_recall_score']]

pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('expand_frame_repr', False)

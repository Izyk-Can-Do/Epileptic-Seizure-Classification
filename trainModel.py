import os
import joblib
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, DetCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve

#################### Custom Functions ####################


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(16, 8), dpi=600)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("AUC")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="b")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="b",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def calc_prevalence(seizure_check1):
    return sum(seizure_check1) / len(seizure_check1)


def calc_specificity(y_actual, y_pred, thresh):
    return sum((y_pred < thresh) & (y_actual == 0)) / sum(y_actual == 0)


def print_report(y_actual, y_pred, thresh):
    auc = roc_auc_score(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, (y_pred > thresh))
    recall = recall_score(y_actual, (y_pred > thresh))
    precision = precision_score(y_actual, (y_pred > thresh))
    specificity = calc_specificity(y_actual, y_pred, thresh)
    print('AUC:%.3f' % auc)
    print('accuracy:%.3f' % accuracy)
    print('recall:%.3f' % recall)
    print('precision:%.3f' % precision)
    print('specificity:%.3f' % specificity)
    print('prevalence:%.3f' % calc_prevalence(y_actual))
    print(' ')
    return auc, accuracy, recall, precision, specificity


# Data Reading
train_chb = 'chb05/'
train_csv_index = 1
dir_path = os.path.dirname(os.path.realpath(__file__))
read_path = dir_path + '/EDF Files/' + train_chb
edf_files = sorted([file for file in os.listdir(read_path) if file.endswith('.edf')])
train_csv = edf_files[train_csv_index].replace('.edf', '')
save_path = 'Figures/Model Figures/%s/%s' % (train_chb, train_csv)
os.makedirs(save_path) if not os.path.exists(save_path) else False
csv_file = '%s/CSV Files/%s/%s.csv' % (dir_path, train_chb, train_csv)
df = pd.read_csv(csv_file)
warnings.filterwarnings('ignore')

# Building Train, Test and Validation Sets
collist = df.columns.tolist()
col_count = df.shape[1]
cols_input = collist[1:col_count]
data = df.values
X, y = data[:, :-1], data[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
thresh = 0.5

# Pre-processing
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)

#################### Model: K nearest neighbors (KNN) ####################
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

y_train_preds = knn.predict_proba(X_train)[:, 1]
y_valid_preds = knn.predict_proba(X_valid)[:, 1]

print('KNN')
print('Training:')
knn_train_auc, knn_train_accuracy, knn_train_recall, \
    knn_train_precision, knn_train_specificity = print_report(y_train, y_train_preds, thresh)
print('Validation:')
knn_valid_auc, knn_valid_accuracy, knn_valid_recall, \
    knn_valid_precision, knn_valid_specificity = print_report(y_valid, y_valid_preds, thresh)

#################### Model: Logistic Regression ####################
lr = LogisticRegression(random_state=69)
lr.fit(X_train, y_train)

y_train_preds = lr.predict_proba(X_train)[:, 1]
y_valid_preds = lr.predict_proba(X_valid)[:, 1]

print('Logistic Regression')
print('Training:')
lr_train_auc, lr_train_accuracy, lr_train_recall, \
    lr_train_precision, lr_train_specificity = print_report(y_train, y_train_preds, thresh)
print('Validation:')
lr_valid_auc, lr_valid_accuracy, lr_valid_recall, \
    lr_valid_precision, lr_valid_specificity = print_report(y_valid, y_valid_preds, thresh)

#################### Model: Stochastic Gradient Descent ####################
sgdc = SGDClassifier(loss='log', alpha=0.1)
sgdc.fit(X_train, y_train)

y_train_preds = sgdc.predict_proba(X_train)[:, 1]
y_valid_preds = sgdc.predict_proba(X_valid)[:, 1]

print('sgdc')
print('Training:')
sgdc_train_auc, sgdc_train_accuracy, sgdc_train_recall, \
    sgdc_train_precision, sgdc_train_specificity = print_report(y_train, y_train_preds, thresh)
print('Validation:')
sgdc_valid_auc, sgdc_valid_accuracy, sgdc_valid_recall, \
    sgdc_valid_precision, sgdc_valid_specificity = print_report(y_valid, y_valid_preds, thresh)

#################### Model: Naive Bayes ####################
nb = GaussianNB()
nb.fit(X_train, y_train)

y_train_preds = nb.predict_proba(X_train)[:, 1]
y_valid_preds = nb.predict_proba(X_valid)[:, 1]

print('Naive Bayes')
print('Training:')
nb_train_auc, nb_train_accuracy, nb_train_recall, nb_train_precision, \
    nb_train_specificity = print_report(y_train, y_train_preds, thresh)
print('Validation:')
nb_valid_auc, nb_valid_accuracy, nb_valid_recall, nb_valid_precision, \
    nb_valid_specificity = print_report(y_valid, y_valid_preds, thresh)

#################### Model: Decision Tree Classifier ####################
tree = DecisionTreeClassifier(max_depth=10, random_state=69)
tree.fit(X_train, y_train)
y_train_preds = tree.predict_proba(X_train)[:, 1]
y_valid_preds = tree.predict_proba(X_valid)[:, 1]

print('Decision Tree')
print('Training:')
tree_train_auc, tree_train_accuracy, tree_train_recall, tree_train_precision, \
    tree_train_specificity = print_report(y_train, y_train_preds, thresh)
print('Validation:')
tree_valid_auc, tree_valid_accuracy, tree_valid_recall, tree_valid_precision, \
    tree_valid_specificity = print_report(y_valid, y_valid_preds, thresh)

#################### Model: Random Forest ####################
rf = RandomForestClassifier(max_depth=6, random_state=69)
rf.fit(X_train, y_train)

y_train_preds = rf.predict_proba(X_train)[:, 1]
y_valid_preds = rf.predict_proba(X_valid)[:, 1]

print('Random Forest')
print('Training:')
rf_train_auc, rf_train_accuracy, rf_train_recall, rf_train_precision, \
    rf_train_specificity = print_report(y_train, y_train_preds, thresh)
print('Validation:')
rf_valid_auc, rf_valid_accuracy, rf_valid_recall, rf_valid_precision, \
    rf_valid_specificity = print_report(y_valid, y_valid_preds, thresh)

#################### Model: Gradient Boosting Classifier ####################
gbc = GradientBoostingClassifier(
    n_estimators=100, learning_rate=1.0, max_depth=3, random_state=69)
gbc.fit(X_train, y_train)

y_train_preds = gbc.predict_proba(X_train)[:, 1]
y_valid_preds = gbc.predict_proba(X_valid)[:, 1]

print('Gradient Boosting Classifier')
print('Training:')
gbc_train_auc, gbc_train_accuracy, gbc_train_recall, gbc_train_precision, \
    gbc_train_specificity = print_report(y_train, y_train_preds, thresh)
print('Validation:')
gbc_valid_auc, gbc_valid_accuracy, gbc_valid_recall, gbc_valid_precision, \
    gbc_valid_specificity = print_report(y_valid, y_valid_preds, thresh)

#################### Model: XGBoost Classifier ####################
xgbc = XGBClassifier(eval_metric='auc')
xgbc.fit(X_train, y_train)

y_train_preds = xgbc.predict_proba(X_train)[:, 1]
y_valid_preds = xgbc.predict_proba(X_valid)[:, 1]

print('Xtreme Gradient Boosting Classifier')
print('Training:')
xgbc_train_auc, xgbc_train_accuracy, xgbc_train_recall, xgbc_train_precision, \
    xgbc_train_specificity = print_report(y_train, y_train_preds, thresh)
print('Validation:')
xgbc_valid_auc, xgbc_valid_accuracy, xgbc_valid_recall, xgbc_valid_precision, \
    xgbc_valid_specificity = print_report(y_valid, y_valid_preds, thresh)

#################### Model: Multi-layer Perceptron ####################
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
mlp.fit(X_train, y_train)

y_train_preds = mlp.predict_proba(X_train)[:, 1]
y_valid_preds = mlp.predict_proba(X_valid)[:, 1]

print('Multi-layer Perceptron')
print('Training:')
mlp_train_auc, mlp_train_accuracy, mlp_train_recall, mlp_train_precision, \
    mlp_train_specificity = print_report(y_train, y_train_preds, thresh)
print('Validation:')
mlp_valid_auc, mlp_valid_accuracy, mlp_valid_recall, mlp_valid_precision, \
    mlp_valid_specificity = print_report(y_valid, y_valid_preds, thresh)

#################### Model: Extremely Random Trees ####################
etc = ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=1.0,
                           min_samples_leaf=3, min_samples_split=20, n_estimators=100)
etc.fit(X_train, y_train)

y_train_preds = etc.predict_proba(X_train)[:, 1]
y_valid_preds = etc.predict_proba(X_valid)[:, 1]

print('Extremely Randomized Trees Classifier')
print('Training:')
etc_train_auc, etc_train_accuracy, etc_train_recall, etc_train_precision, \
    etc_train_specificity = print_report(y_train, y_train_preds, thresh)
print('Validation:')
etc_valid_auc, etc_valid_accuracy, etc_valid_recall, etc_valid_precision, \
    etc_valid_specificity = print_report(y_valid, y_valid_preds, thresh)

#################### Evaluation ####################
final_model = etc
model_name = 'Epileptic Seizure Detection Model.pkl'
joblib.dump(final_model, model_name)

y_train_preds = final_model.predict_proba(X_train)[:, 1]
y_valid_preds = final_model.predict_proba(X_valid)[:, 1]
y_test_preds = final_model.predict_proba(X_test)[:, 1]

print('Test:')
test_auc, test_accuracy, test_recall, test_precision,  \
    test_specificity = print_report(y_test, y_test_preds, thresh)

#################### Figure: Confusion Matrix For Evaluation ####################
predictions_evl = final_model.predict(X_test)
cm = confusion_matrix(y_test, predictions_evl, labels=final_model.classes_)
disp_confm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=final_model.classes_)
plt.figure(figsize=(15, 7), dpi=600)
ax_cm = plt.gca()
disp_confm.plot(ax=ax_cm)
plt.tick_params(axis='both', labelsize=12, pad=5)
plt.title('Confusion Matrix for Evaluation')
plt.savefig('%s/%s - Confusion Matrix for Evaluation.png' % (save_path, train_csv))

#################### Figure: Training & Validation AUC Results ####################
df_results = pd.DataFrame({
    'classifier':
    ['KNN', 'KNN', 'LR', 'LR', 'SGDC', 'SGDC', 'NB', 'NB', 'DT', 'DT', 'RF', 'RF', 'GBC', 'GBC',
     'XGBC', 'XGBC', 'MLP', 'MLP', 'ETC', 'ETC'],
    'data_set': ['train', 'valid'] * 10,
    'auc':
    [knn_train_auc, knn_valid_auc, lr_train_auc, lr_valid_auc, sgdc_train_auc, sgdc_valid_auc,
     nb_train_auc, nb_valid_auc, tree_train_auc, tree_valid_auc, rf_train_auc, rf_valid_auc,
     gbc_train_auc, gbc_valid_auc, xgbc_train_auc, xgbc_valid_auc, mlp_train_auc, mlp_valid_auc,
     etc_train_auc, etc_valid_auc]})

sns.set(style="whitegrid")
sns.set_style("whitegrid")
plt.figure(figsize=(16, 6), dpi=600)
ax = sns.barplot(x='classifier', y='auc', hue='data_set', data=df_results, palette='flare')
ax.set_xlabel('Classifier', fontsize=12)
ax.set_ylabel('AUC', fontsize=12)
ax.tick_params(labelsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
plt.tick_params(axis='both', labelsize=12, pad=5)
plt.title('Training & Validation AUC Results')
plt.savefig('%s/%s - Training & Validation AUC Results.png' % (save_path, train_csv))

#################### Figure: Training & Validation Accuracy Results ####################
df_results = pd.DataFrame({
    'classifier':
    ['KNN', 'KNN', 'LR', 'LR', 'SGDC', 'SGDC', 'NB', 'NB', 'DT', 'DT', 'RF', 'RF', 'GBC', 'GBC',
     'XGBC', 'XGBC', 'MLP', 'MLP', 'ETC', 'ETC'],
    'data_set': ['train', 'valid'] * 10,
    'accuracy':
    [knn_train_accuracy, knn_valid_accuracy, lr_train_accuracy, lr_valid_accuracy,
     sgdc_train_accuracy, sgdc_valid_accuracy, nb_train_accuracy, nb_valid_accuracy,
     tree_train_accuracy, tree_valid_accuracy, rf_train_accuracy, rf_valid_accuracy,
     gbc_train_accuracy, gbc_valid_accuracy, xgbc_train_accuracy, xgbc_valid_accuracy,
     mlp_train_accuracy, mlp_valid_accuracy, etc_train_accuracy, etc_valid_accuracy]})

sns.set(style="whitegrid")
sns.set_style("whitegrid")
plt.figure(figsize=(16, 6), dpi=600)
ax = sns.barplot(x='classifier', y='accuracy', hue='data_set', data=df_results, palette='flare')
ax.set_xlabel('Classifier', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.tick_params(labelsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
plt.tick_params(axis='both', labelsize=12, pad=5)
plt.title('Training & Validation Accuracy Results')
plt.savefig('%s/%s - Training & Validation Accuracy Results.png' % (save_path, train_csv))

#################### Figure: Training & Validation Recall Results ####################
df_results = pd.DataFrame({
    'classifier':
    ['KNN', 'KNN', 'LR', 'LR', 'SGDC', 'SGDC', 'NB', 'NB', 'DT', 'DT', 'RF', 'RF', 'GBC', 'GBC',
     'XGBC', 'XGBC', 'MLP', 'MLP', 'ETC', 'ETC'],
    'data_set': ['train', 'valid'] * 10,
    'recall':
    [knn_train_recall, knn_valid_recall, lr_train_recall, lr_valid_recall, sgdc_train_recall,
     sgdc_valid_recall, nb_train_recall, nb_valid_recall, tree_train_recall, tree_valid_recall,
     rf_train_recall, rf_valid_recall, gbc_train_recall, xgbc_valid_recall, xgbc_train_recall,
     xgbc_valid_recall, mlp_train_recall, mlp_valid_recall, etc_train_recall, etc_valid_recall]})

sns.set(style="whitegrid")
sns.set_style("whitegrid")
plt.figure(figsize=(16, 6), dpi=600)
ax = sns.barplot(x='classifier', y='recall', hue='data_set', data=df_results, palette='flare')
ax.set_xlabel('Classifier', fontsize=12)
ax.set_ylabel('Recall', fontsize=12)
ax.tick_params(labelsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
plt.tick_params(axis='both', labelsize=12, pad=5)
plt.title('Training & Validation Recall Results')
plt.savefig('%s/%s - Training & Validation Recall Results.png' % (save_path, train_csv))

#################### Figure: Training & Validation Precision Results ####################
df_results = pd.DataFrame({
    'classifier':
    ['KNN', 'KNN', 'LR', 'LR', 'SGDC', 'SGDC', 'NB', 'NB', 'DT', 'DT', 'RF', 'RF', 'GBC', 'GBC',
     'XGBC', 'XGBC', 'MLP', 'MLP', 'ETC', 'ETC'],
    'data_set': ['train', 'valid'] * 10,
    'precision':
    [knn_train_precision, knn_valid_precision, lr_train_precision, lr_valid_precision,
     sgdc_train_precision, sgdc_valid_precision, nb_train_precision, nb_valid_precision,
     tree_train_precision, tree_valid_precision, rf_train_precision, rf_valid_precision,
     gbc_train_precision, gbc_valid_precision, xgbc_train_precision, xgbc_valid_precision,
     mlp_train_precision, mlp_valid_precision, etc_train_precision, etc_valid_precision]})

sns.set(style="whitegrid")
sns.set_style("whitegrid")
plt.figure(figsize=(16, 6), dpi=600)
ax = sns.barplot(x='classifier', y='precision', hue='data_set', data=df_results, palette='flare')
ax.set_xlabel('Classifier', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.tick_params(labelsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
plt.tick_params(axis='both', labelsize=12, pad=5)
plt.title('Training & Validation Precision Results')
plt.savefig('%s/%s - Training & Validation Precision Results.png' % (save_path, train_csv))

#################### Figure: Training & Validation Specificity Results ####################
df_results = pd.DataFrame({
    'classifier':
    ['KNN', 'KNN', 'LR', 'LR', 'SGDC', 'SGDC', 'NB', 'NB', 'DT', 'DT', 'RF', 'RF', 'GBC', 'GBC',
     'XGBC', 'XGBC', 'MLP', 'MLP', 'ETC', 'ETC'],
    'data_set': ['train', 'valid'] * 10,
    'specificity':
    [knn_train_specificity, knn_valid_specificity, lr_train_specificity, lr_valid_specificity,
     sgdc_train_specificity, sgdc_valid_specificity, nb_train_specificity, nb_valid_specificity,
     tree_train_specificity, tree_valid_specificity, rf_train_specificity, rf_valid_specificity,
     gbc_train_specificity, gbc_valid_specificity, xgbc_train_specificity, xgbc_valid_specificity,
     mlp_train_specificity, mlp_valid_specificity, etc_train_specificity, etc_valid_specificity]})

sns.set(style="whitegrid")
sns.set_style("whitegrid")
plt.figure(figsize=(16, 6), dpi=600)
ax = sns.barplot(x='classifier', y='specificity', hue='data_set', data=df_results, palette='flare')
ax.set_xlabel('Classifier', fontsize=12)
ax.set_ylabel('Specificity', fontsize=12)
ax.tick_params(labelsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
plt.tick_params(axis='both', labelsize=12, pad=5)
plt.title('Training & Validation Specificity Results')
plt.savefig('%s/%s - Training & Validation Specificity Results.png' % (save_path, train_csv))

#################### Figure: Learning Curve For Model ####################
title = 'AUC Learning Curve for ExtraTrees'
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
estimator = ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=1.0,
                                 min_samples_leaf=3, min_samples_split=20, n_estimators=100)
plot_learning_curve(estimator, title, X_train, y_train, ylim=(0.2, 1.01), cv=cv, n_jobs=4)
plt.tick_params(axis='both', labelsize=12, pad=5)
plt.title('Learning Curve for Model')
plt.savefig('%s/%s - Learning Curve for Model.png' % (save_path, train_csv))

#################### Figure: ROC Curve For Model ####################
fpr_train, tpr_train, thresholds_test = roc_curve(y_train, y_train_preds)
auc_train = roc_auc_score(y_train, y_train_preds)

fpr_valid, tpr_valid, thresholds_valid = roc_curve(y_valid, y_valid_preds)
auc_valid = roc_auc_score(y_valid, y_valid_preds)

fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_preds)
auc_test = roc_auc_score(y_test, y_test_preds)

plt.figure(figsize=(15, 7), dpi=600)
plt.plot(fpr_train, tpr_train, 'r-', label='Train AUC:%.3f' % auc_train)
plt.plot(fpr_valid, tpr_valid, 'b-', label='Valid AUC:%.3f' % auc_valid)
plt.plot(fpr_test, tpr_test, 'g-', label='Test AUC:%.3f' % auc_test)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.tick_params(axis='both', labelsize=12, pad=5)
plt.title('ROC Curve for Model')
plt.legend()
plt.savefig('%s/%s - ROC Curve for Model.png' % (save_path, train_csv))

#################### Figure: ROC Curve For All Models ####################
knn_roc = RocCurveDisplay.from_estimator(knn, X_train, y_train)
fig, ax_roc = plt.subplots(figsize=(16, 12), dpi=300)
lr_roc = RocCurveDisplay.from_estimator(lr,  X_train, y_train, ax=ax_roc, alpha=0.8)
sgdc_roc = RocCurveDisplay.from_estimator(sgdc, X_train, y_train, ax=ax_roc, alpha=0.8)
nb_roc = RocCurveDisplay.from_estimator(nb, X_train, y_train, ax=ax_roc, alpha=0.8)
dt_roc = RocCurveDisplay.from_estimator(tree, X_train, y_train, ax=ax_roc, alpha=0.8)
rf_roc = RocCurveDisplay.from_estimator(rf, X_train, y_train, ax=ax_roc, alpha=0.8)
gbc_roc = RocCurveDisplay.from_estimator(gbc, X_train, y_train, ax=ax_roc, alpha=0.8)
xgbc_roc = RocCurveDisplay.from_estimator(xgbc, X_train, y_train, ax=ax_roc, alpha=0.8)
mlp_roc = RocCurveDisplay.from_estimator(mlp, X_train, y_train, ax=ax_roc, alpha=0.8)
etc_roc = RocCurveDisplay.from_estimator(etc, X_train, y_train, ax=ax_roc, alpha=0.8)
knn_roc.plot(ax=ax_roc, alpha=0.8,)
plt.tick_params(axis='both', labelsize=12, pad=5)
plt.title('ROC Curve for All Models')
plt.savefig('%s/%s - ROC Curve for All Models.png' % (save_path, train_csv))

#################### Figure: Positive Feature Importances ####################
feature_importances = pd.DataFrame(final_model.feature_importances_, index=cols_input, columns=[
                                   'importance']).sort_values('importance', ascending=False)
pos_features = feature_importances.loc[feature_importances.importance > 0]
num = np.min([50, len(pos_features)])
ylocs = np.arange(num)
values_to_plot = pos_features.iloc[:num].values.ravel()[::-1]
feature_labels = list(pos_features.iloc[:num].index)[::-1]
plt.figure(num=None, figsize=(20, 15), dpi=600, facecolor='w', edgecolor='k')
plt.barh(ylocs, values_to_plot, align='center')
plt.ylabel('Features')
plt.xlabel('Importance Score')
plt.tick_params(axis='both', labelsize=12, pad=5)
plt.title('Positive Feature Importances for Model')
plt.yticks(ylocs, feature_labels)
plt.savefig('%s/%s - Positive Feature Importances.png' % (save_path, train_csv))

#################### Figure: Negative Feature Importances ####################
feature_importances = pd.DataFrame(final_model.feature_importances_, index=cols_input, columns=[
                                   'importance']).sort_values('importance', ascending=False)
neg_features = feature_importances.loc[feature_importances.importance < 0]
if len(neg_features) > 0:
    num = np.min([50, len(neg_features)])
    ylocs = np.arange(num)
    values_to_plot = neg_features.iloc[:num].values.ravel()[::-1]
    feature_labels = list(neg_features.iloc[:num].index)[::-1]
    plt.figure(num=None, figsize=(20, 15), dpi=600, facecolor='w', edgecolor='k')
    plt.barh(ylocs, values_to_plot, align='center')
    plt.ylabel('Features')
    plt.xlabel('Importance Score')
    plt.tick_params(axis='both', labelsize=12, pad=5)
    plt.title('Negative Feature Importances for Model')
    plt.yticks(ylocs, feature_labels)
    plt.savefig('%s/%s - Negative Feature Importances.png' % (save_path, train_csv))

#################### Figure: Test Results For Model ####################
df_results = pd.DataFrame({
    'data_set': ['Validation Accuracy', 'Test Accuracy'],
    'test_reulsts': [etc_valid_accuracy, test_accuracy]})

sns.set(style="whitegrid")
sns.set_style("whitegrid")
plt.figure(figsize=(8, 12), dpi=600)
bars = plt.bar(df_results['data_set'], df_results['test_reulsts'])
bars[0].set_color('r')
plt.xlabel('Accuracy', fontsize=12)
plt.ylabel('Scores', fontsize=12)
plt.tick_params(labelsize=12)
plt.tick_params(axis='both', labelsize=12, pad=5)
plt.title('Validation and Test Result for Model')
plt.savefig('%s/%s - Validation and Test Result.png' % (save_path, train_csv))

#################### Figure: Calibration Curve For Model ####################
disp_cal = CalibrationDisplay.from_predictions(y_test, y_test_preds)
plt.figure(figsize=(15, 7), dpi=600)
ax_cd = plt.gca()
disp_cal.plot(ax=ax_cd)
plt.tick_params(axis='both', labelsize=12, pad=5)
plt.title('Calibration Curve for Model')
plt.savefig('%s/%s - Calibration Curve.png' % (save_path, train_csv))

#################### Figure: DET Curve For Model ####################
disp_det = DetCurveDisplay.from_predictions(y_test, y_test_preds)
plt.figure(figsize=(15, 7), dpi=600)
ax_det = plt.gca()
disp_det.plot(ax=ax_det)
plt.tick_params(axis='both', labelsize=12, pad=5)
plt.title('DET Curve for Model')
plt.savefig('%s/%s - DET Curve.png' % (save_path, train_csv))

#################### Figure: Precision Recall Display For Model ####################
disp_prec = PrecisionRecallDisplay.from_predictions(y_test, y_test_preds)
plt.figure(figsize=(15, 7), dpi=600)
ax_prec = plt.gca()
disp_prec.plot(ax=ax_prec)
plt.tick_params(axis='both', labelsize=12, pad=5)
plt.title('Precision & Recall for Model')
plt.savefig('%s/%s - Precision & Recall.png' % (save_path, train_csv))

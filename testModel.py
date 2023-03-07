import os
import re
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import DetCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, roc_curve, confusion_matrix

#################### Custom Functions ####################


def calc_prevalence(seizure_check1):
    return sum(seizure_check1) / len(seizure_check1)


def calc_specificity(y_actual, y_pred, thresh):
    return sum((y_pred < thresh) & (y_actual == 0)) / sum(y_actual == 0)


def print_report(y_actual, y_pred, thresh):
    auc = roc_auc_score(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, (y_pred > thresh))
    print('AUC:%.3f' % auc)
    print('accuracy:%.3f' % accuracy)
    print(' ')
    return auc, accuracy

#################### Informatipon Data Reading ####################
test_chb = 'chb15/'
test_csv_index = 0
dir_path = os.path.dirname(os.path.realpath(__file__))
read_path = dir_path + '/EDF Files/' + test_chb
edf_files = sorted([file for file in os.listdir(read_path) if file.endswith('.edf')])
test_csv = edf_files[test_csv_index].replace('.edf', '')
save_path = 'Figures/Patient Figures/%s/%s' % (test_chb, test_csv)
os.makedirs(save_path) if not os.path.exists(save_path) else False
csv_file = '%s/CSV Files/%s%s.csv' % (dir_path, test_chb, test_csv)
txt_file = [file for file in os.listdir(read_path) if file.endswith('.txt')]
summary_file = open(read_path + txt_file[0], 'r')
summary_text = summary_file.read()
seizure_times = np.asarray(re.findall('(?<=Number of Seizures in File: )((?!0)\d{1})', summary_text)).astype(np.int64)
start_times = np.asarray(re.findall('(?<=Start Time: )(.*?)(?= sec)', summary_text)).astype(np.int64)
end_times = np.asarray(re.findall('(?<=End Time: )(.*?)(?= sec)', summary_text)).astype(np.int64)
summary_file.close()
df_test = pd.read_csv(csv_file)

#################### Model Reading ####################
model_name = 'Epileptic Seizure Detection Model.pkl'
etc = joblib.load(model_name)
thresh = 0.5

collist = df_test.columns.tolist()
col_count = df_test.shape[1]
cols_input = collist[1:col_count]
X_test = df_test[cols_input].values
scaler = StandardScaler()
scaler.fit(X_test)
X_test = scaler.transform(X_test)
y_test = df_test['seizure_check'].values
y_test_preds = etc.predict_proba(X_test)[:, 1]
predictions_test = etc.predict(X_test)

print('List of edf files: %s' % edf_files)
print('Testing %s\n' % edf_files[test_csv_index])

print('Test:')
etc_test_auc, etc_test_accuracy = print_report(y_test, y_test_preds, thresh)

#################### Figure: Confusion Matrix For Testing ####################
cm = confusion_matrix(y_test, predictions_test, labels=etc.classes_)
disp_confm_new = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=etc.classes_)
plt.figure(figsize=(15, 7), dpi=600)
ax_cmt = plt.gca()
disp_confm_new.plot(ax=ax_cmt)
plt.tick_params(axis='both', labelsize=12, pad=5)
plt.title('Confusion Matrix for Testing %s' % test_csv)
plt.savefig('%s/%s - Confusion Matrix.png' % (save_path, test_csv))

#################### Figure: Test Results For New Signal ####################
df_results = pd.DataFrame({
    'data_set': ['AUC', 'Accuracy'],
    'test_reulsts': [etc_test_auc, etc_test_accuracy]})

sns.set(style="whitegrid")
sns.set_style("whitegrid")
plt.figure(figsize=(8, 12), dpi=600)
bars = plt.bar(df_results['data_set'], df_results['test_reulsts'])
bars[0].set_color('r')
plt.xlabel('AUC and Accuracy', fontsize=12)
plt.ylabel('Scores', fontsize=12)
plt.tick_params(labelsize=12)
plt.tick_params(axis='both', labelsize=12, pad=5)
plt.title('Test Result for %s' % test_csv)
plt.savefig('%s/%s - Test Result.png' % (save_path, test_csv))

#################### Figure: ROC Curve For New Signal ####################
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_preds)
auc_test = roc_auc_score(y_test, y_test_preds)

plt.figure(figsize=(16, 10),dpi=600)
plt.plot(fpr_test, tpr_test, 'g-', label='Test AUC:%.3f' % auc_test)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.tick_params(axis='both', labelsize=12, pad=5)
plt.title('ROC Curve for %s' % test_csv)
plt.legend()
plt.savefig('%s/%s - ROC Curve.png' % (save_path, test_csv))

#################### Figure: Calibration Curve For New Signal ####################
disp_cal = CalibrationDisplay.from_predictions(y_test, y_test_preds)
plt.figure(figsize=(15, 7), dpi=600)
ax_cd = plt.gca()
disp_cal.plot(ax=ax_cd)
plt.tick_params(axis='both', labelsize=12, pad=5)
plt.title('Calibration Curve for %s' % test_csv)
plt.savefig('%s/%s - Calibration Curve.png' % (save_path, test_csv))

#################### Figure: DET Curve For New Signal ####################
disp_det = DetCurveDisplay.from_predictions(y_test, y_test_preds)
plt.figure(figsize=(15, 7), dpi=600)
ax_det = plt.gca()
disp_det.plot(ax=ax_det)
plt.tick_params(axis='both', labelsize=12, pad=5)
plt.title('DET Curve for %s' % test_csv)
plt.savefig('%s/%s - DET Curve.png' % (save_path, test_csv))

#################### Figure: Precision & Recall For New Signal ####################
disp_prec = PrecisionRecallDisplay.from_predictions(y_test, y_test_preds)
plt.figure(figsize=(15, 7), dpi=600)
ax_prec = plt.gca()
disp_prec.plot(ax=ax_prec)
plt.tick_params(axis='both', labelsize=12, pad=5)
plt.title('Precision & Recall for %s' % test_csv)
plt.savefig('%s/%s - Precision & Recall.png' % (save_path, test_csv))

#################### Figure: Classification Probability For New Signal ####################
plt.figure(figsize=(15, 7), dpi=600)
plt.hist(y_test_preds[y_test == 0], bins=50, label='Negatives')
plt.hist(y_test_preds[y_test == 1], bins=50, label='Positives', alpha=0.7, color='r')
plt.xlabel('Probability of being Positive Class', fontsize=12)
plt.ylabel('Number of Records in each Bucket', fontsize=12)
plt.legend(fontsize=12)
plt.tick_params(axis='both', labelsize=12, pad=5)
plt.title('Classification Probability for %s' % test_csv)
plt.savefig('%s/%s - Classification Probability.png' % (save_path, test_csv))

#################### Figure: Probability with Time Domain For New Signal ####################
plt.figure(figsize=(15, 7), dpi=600)
time = range(len(y_test_preds))
plt.plot(time, y_test_preds)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Seizure Probability', fontsize=12)
plt.tick_params(axis='both', labelsize=12, pad=5)
plt.title('Seizure Probability for %s' % test_csv)
plt.savefig('%s/%s - Seizure Probability.png' % (save_path, test_csv))

#################### Figure: Predictions with Time Domain For New Signal ####################
plt.figure(figsize=(15, 7), dpi=600)
time = range(len(predictions_test))
plt.plot(time, predictions_test)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Seizure Prediction', fontsize=12)
plt.tick_params(axis='both', labelsize=12, pad=5)
plt.title('Seizure Prediction for %s' % test_csv)
plt.savefig('%s/%s - Seizure Prediction.png' % (save_path, test_csv))
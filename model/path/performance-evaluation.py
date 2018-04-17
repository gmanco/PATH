import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


loss_file_path = 'PATH TO THE FILE IN THE DIRECTORY results/DATASET_NAME'
#
#
df = pd.read_csv(loss_file_path,sep='\t')
expected_prob = df['prob_active']

y_true = df['active']

only_active=y_true==1
d=expected_prob[only_active]


lkl=np.sum(np.log(d))
print("lkl=",lkl)


y_probas = expected_prob#
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probas)
# Print ROC curve
plt.plot(fpr,tpr)
plt.show()
# Print AUC
auc = np.trapz(tpr,fpr)
print('AUC :', auc)


from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_true, y_probas)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))


precision, recall, _ = metrics.precision_recall_curve(y_true, y_probas)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))

plt.show()
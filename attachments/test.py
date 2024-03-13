import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from joblib import load



def evaluate_model(model, test_data):
    predictions = model.predict(test_data)
    y_pred = np.round(predictions).astype(int)
    y_true = test_data.labels

    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


model = load('model.joblib')
evaluate_model(model, "/test_videos")

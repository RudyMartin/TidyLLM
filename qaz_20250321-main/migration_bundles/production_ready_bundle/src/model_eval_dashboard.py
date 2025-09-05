
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

def compute_topic_accuracy(df, label_col='ground_truth', pred_col='prediction', topic_col='topic'):
    """Compute accuracy per topic."""
    topic_scores = (
        df.groupby(topic_col)
        .apply(lambda g: accuracy_score(g[label_col], g[pred_col]))
        .reset_index(name='accuracy')
    )
    return topic_scores

def plot_topic_accuracy(topic_scores):
    """Plot accuracy per topic as a horizontal bar chart."""
    plt.figure(figsize=(10, 6))
    sns.barplot(data=topic_scores, x='accuracy', y='topic')
    plt.title('Accuracy by Topic')
    plt.xlabel('Accuracy')
    plt.ylabel('Topic')
    plt.tight_layout()
    return plt.gcf()

def compute_confusion(df, label_col='ground_truth', pred_col='prediction'):
    """Compute confusion matrix."""
    y_true = df[label_col]
    y_pred = df[pred_col]
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm, labels

def plot_confusion_matrix(cm, labels):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title('Confusion Matrix')
    return fig

def find_topic_winners(df, label_col='ground_truth', pred_col='prediction', topic_col='topic', model_col='model'):
    """Determine which model performs best for each topic."""
    grouped = df.groupby([topic_col, model_col]).apply(
        lambda g: accuracy_score(g[label_col], g[pred_col])
    ).reset_index(name='accuracy')
    best_by_topic = grouped.loc[grouped.groupby(topic_col)['accuracy'].idxmax()].reset_index(drop=True)
    return best_by_topic

# ============================================================
# utils/metrics.py
# PURPOSE: Calculate all evaluation metrics and generate
#          all plots — confusion matrix, ROC curves,
#          training history, F1 bar chart, evaluation report
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import datetime

# Import settings from config.py
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ============================================================
# FUNCTION 1 — compute_all_metrics()
# Calculates every metric and prints them clearly
# ============================================================
def compute_all_metrics(y_true, y_pred_probs, class_names):
    """
    Computes all evaluation metrics.

    y_true       : true class indices  e.g. [0, 4, 2, 6, 1...]
    y_pred_probs : model output probabilities, shape (n_samples, 7)
    class_names  : list of class names e.g. ['akiec','bcc',...]

    Returns: dictionary containing all computed metrics
    """

    print("\n" + "="*55)
    print("  COMPUTING EVALUATION METRICS")
    print("="*55)

    # --- GET PREDICTED CLASS INDICES ---
    # argmax finds the index of highest probability
    # Example: [0.01, 0.02, 0.03, 0.05, 0.87, 0.01, 0.01]
    #          argmax = 4 (mel has highest probability 0.87)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # --- OVERALL ACCURACY ---
    # Fraction of all predictions that were correct
    accuracy = np.mean(y_pred == y_true)
    print(f"\n📊 Overall Accuracy : {accuracy*100:.2f}%")

    # --- CLASSIFICATION REPORT ---
    # Shows precision, recall, F1 for EACH class separately
    print(f"\n📋 Per-Class Report:")
    print("-"*55)
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True   # returns dictionary so we can store values
    )
    # Also print human-readable version
    print(classification_report(
        y_true, y_pred,
        target_names=class_names
    ))

    # --- ROC-AUC SCORE ---
    # Binarize labels for one-vs-rest ROC calculation
    # Example: class 4 (mel) → [0,0,0,0,1,0,0]
    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))

    # Macro average AUC — treats all classes equally
    try:
        auc_macro = roc_auc_score(
            y_true_bin, y_pred_probs,
            multi_class='ovr',
            average='macro'
        )
        print(f"📈 ROC-AUC Score (macro) : {auc_macro:.4f}")
    except Exception as e:
        auc_macro = 0.0
        print(f"⚠️  ROC-AUC could not be computed: {e}")

    # --- BUILD RESULTS DICTIONARY ---
    # Store everything in one dictionary to return
    metrics_dict = {
        'accuracy'          : accuracy,
        'auc_macro'         : auc_macro,
        'report'            : report,
        'y_pred'            : y_pred,
        'y_true'            : y_true,
        'y_pred_probs'      : y_pred_probs,
        'y_true_bin'        : y_true_bin,
        'weighted_f1'       : report['weighted avg']['f1-score'],
        'macro_f1'          : report['macro avg']['f1-score'],
        'weighted_precision': report['weighted avg']['precision'],
        'weighted_recall'   : report['weighted avg']['recall'],
    }

    print(f"\n🏆 Summary:")
    print(f"   Accuracy          : {accuracy*100:.2f}%")
    print(f"   Weighted F1-Score : {metrics_dict['weighted_f1']:.4f}")
    print(f"   Macro F1-Score    : {metrics_dict['macro_f1']:.4f}")
    print(f"   ROC-AUC (macro)   : {auc_macro:.4f}")

    return metrics_dict


# ============================================================
# FUNCTION 2 — plot_confusion_matrix()
# Creates and saves a heatmap of the confusion matrix
# ============================================================
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Plots a normalized confusion matrix as a colour heatmap.

    Normalized means values are 0.0 to 1.0 (proportions)
    instead of raw counts — easier to compare across classes
    with different numbers of test images.

    Diagonal = correct predictions (want these HIGH)
    Off-diagonal = mistakes (want these LOW)
    """

    print("\n📊 Plotting confusion matrix...")

    # --- COMPUTE CONFUSION MATRIX ---
    cm = confusion_matrix(y_true, y_pred)

    # Normalize: divide each row by its sum
    # So each row shows proportions (0.0 to 1.0) not raw counts
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # --- CREATE FIGURE ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw heatmap using seaborn
    # annot=True shows the number inside each cell
    # fmt='.2f shows 2 decimal places
    # cmap='Blues' uses blue colour scale
    sns.heatmap(
        cm_normalized,
        annot=True,          # show values inside cells
        fmt='.2f',           # 2 decimal places
        cmap='Blues',        # blue colour scheme
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,      # thin lines between cells
        linecolor='gray'
    )

    # Labels and title
    ax.set_title('Normalized Confusion Matrix\n'
                 '(Row = Actual Class, Column = Predicted Class)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Actual Class', fontsize=12)
    ax.set_xlabel('Predicted Class', fontsize=12)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Save the plot
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"✅ Confusion matrix saved to: {save_path}")


# ============================================================
# FUNCTION 3 — plot_roc_curves()
# Plots ROC curve for each class + macro average
# ============================================================
def plot_roc_curves(y_true_bin, y_pred_probs, class_names, save_path):
    """
    Plots one ROC curve per class plus macro average.

    ROC curve shows tradeoff between:
    - True Positive Rate (Recall) on Y axis
    - False Positive Rate on X axis

    AUC (Area Under Curve) shown in legend for each class.
    AUC = 1.0 is perfect. AUC = 0.5 is random guessing.
    """

    print("\n📈 Plotting ROC curves...")

    # Colours for each class curve
    colors = ['#e74c3c', '#3498db', '#2ecc71',
              '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']

    fig, ax = plt.subplots(figsize=(10, 8))

    # --- PLOT ONE CURVE PER CLASS ---
    auc_scores = []
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        # Compute ROC curve for this class vs all others
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])

        # Compute AUC for this class
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)

        # Plot the curve
        ax.plot(
            fpr, tpr,
            color=color,
            linewidth=2,
            label=f'{class_name} (AUC = {roc_auc:.3f})'
        )

    # --- PLOT MACRO AVERAGE ---
    # Average all class ROC curves into one summary curve
    all_fpr = np.unique(np.concatenate(
        [roc_curve(y_true_bin[:, i], y_pred_probs[:, i])[0]
         for i in range(len(class_names))]
    ))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= len(class_names)
    macro_auc = auc(all_fpr, mean_tpr)

    ax.plot(
        all_fpr, mean_tpr,
        color='black',
        linewidth=3,
        linestyle='--',
        label=f'Macro Average (AUC = {macro_auc:.3f})'
    )

    # --- DIAGONAL LINE ---
    # This represents a random classifier (AUC = 0.5)
    # Our model should be well above this line
    ax.plot([0, 1], [0, 1], 'k:', linewidth=1, label='Random Classifier')

    # Labels and formatting
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.set_title('ROC Curves — One vs Rest\n'
                 '(Higher AUC = Better Model)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Save the plot
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"✅ ROC curves saved to: {save_path}")


# ============================================================
# FUNCTION 4 — plot_training_history()
# Plots accuracy and loss curves from training
# ============================================================
def plot_training_history(history_csv_path, save_path):
    """
    Loads training_history.csv and plots:
    - Training vs Validation Accuracy (top graph)
    - Training vs Validation Loss (bottom graph)

    These curves tell you:
    - Is the model learning? (accuracy going up)
    - Is it overfitting? (train accuracy much higher than val accuracy)
    - When did it converge? (curves flatten out)
    """

    print("\n📉 Plotting training history...")

    # Load the CSV saved during training
    history_df = pd.read_csv(history_csv_path)

    # Create figure with 2 subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    epochs = range(1, len(history_df) + 1)

    # --- TOP PLOT: ACCURACY ---
    ax1.plot(epochs, history_df['accuracy'],
             'b-o', linewidth=2, markersize=4, label='Training Accuracy')
    ax1.plot(epochs, history_df['val_accuracy'],
             'r-o', linewidth=2, markersize=4, label='Validation Accuracy')

    # Mark best validation accuracy
    best_epoch = history_df['val_accuracy'].idxmax() + 1
    best_val_acc = history_df['val_accuracy'].max()
    ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7)
    ax1.annotate(f'Best: {best_val_acc:.3f}\n(Epoch {best_epoch})',
                 xy=(best_epoch, best_val_acc),
                 xytext=(best_epoch + 1, best_val_acc - 0.05),
                 fontsize=9, color='green')

    ax1.set_title('Training vs Validation Accuracy',
                  fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # --- BOTTOM PLOT: LOSS ---
    ax2.plot(epochs, history_df['loss'],
             'b-o', linewidth=2, markersize=4, label='Training Loss')
    ax2.plot(epochs, history_df['val_loss'],
             'r-o', linewidth=2, markersize=4, label='Validation Loss')

    ax2.set_title('Training vs Validation Loss',
                  fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Save the plot
    plt.tight_layout(pad=3.0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"✅ Training history plot saved to: {save_path}")


# ============================================================
# FUNCTION 5 — plot_f1_per_class()
# Horizontal bar chart showing F1 score per disease class
# ============================================================
def plot_f1_per_class(metrics_dict, class_names, save_path):
    """
    Creates a horizontal bar chart showing F1-score
    for each of the 7 skin lesion classes.

    Colour coding:
    - Green  : F1 > 0.85 (excellent)
    - Orange : F1 0.70 to 0.85 (acceptable)
    - Red    : F1 < 0.70 (needs improvement)
    """

    print("\n📊 Plotting F1 scores per class...")

    report = metrics_dict['report']

    # Extract F1 score for each class
    f1_scores = []
    for cls in class_names:
        f1 = report[cls]['f1-score']
        f1_scores.append(f1)

    # Assign colour based on F1 value
    colors = []
    for f1 in f1_scores:
        if f1 >= 0.85:
            colors.append('#2ecc71')   # green — excellent
        elif f1 >= 0.70:
            colors.append('#f39c12')   # orange — acceptable
        else:
            colors.append('#e74c3c')   # red — needs improvement

    # --- CREATE HORIZONTAL BAR CHART ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get full class names for display
    full_names = [config.CLASS_FULL_NAMES.get(cls, cls) for cls in class_names]

    bars = ax.barh(full_names, f1_scores, color=colors,
                   edgecolor='white', linewidth=0.5)

    # Add value labels on each bar
    for bar, f1 in zip(bars, f1_scores):
        ax.text(
            bar.get_width() + 0.01,   # x position (just after bar end)
            bar.get_y() + bar.get_height() / 2,  # y position (centre of bar)
            f'{f1:.3f}',              # text to show
            va='center', fontsize=11, fontweight='bold'
        )

    # Add vertical reference lines
    ax.axvline(x=0.85, color='green', linestyle='--',
               alpha=0.5, label='Excellent (0.85)')
    ax.axvline(x=0.70, color='orange', linestyle='--',
               alpha=0.5, label='Acceptable (0.70)')

    ax.set_xlim([0, 1.1])
    ax.set_xlabel('F1-Score', fontsize=12)
    ax.set_title('F1-Score Per Disease Class\n'
                 '(Green=Excellent, Orange=Acceptable, Red=Needs Improvement)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, axis='x', alpha=0.3)

    # Save the plot
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"✅ F1 per class chart saved to: {save_path}")


# ============================================================
# FUNCTION 6 — save_evaluation_report()
# Saves a complete text summary of all results
# ============================================================
def save_evaluation_report(metrics_dict, class_names, save_path):
    """
    Saves a complete evaluation report as a text file.
    This file can be included in your project documentation.
    """

    print("\n📝 Saving evaluation report...")

    report = metrics_dict['report']

    # Build report content as a string
    lines = []
    lines.append("="*60)
    lines.append("  SKIN LESION CLASSIFIER — EVALUATION REPORT")
    lines.append("="*60)
    lines.append(f"  Project : Hybrid Multi-Modal Skin Lesion Classification")
    lines.append(f"  Model   : EfficientNetB0 + Clinical Metadata Fusion")
    lines.append(f"  Dataset : HAM10000 (ISIC 2018) — 7 Classes")
    lines.append(f"  Date    : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("="*60)

    lines.append("\n📊 OVERALL METRICS")
    lines.append("-"*40)
    lines.append(f"  Accuracy          : {metrics_dict['accuracy']*100:.2f}%")
    lines.append(f"  Weighted F1-Score : {metrics_dict['weighted_f1']:.4f}")
    lines.append(f"  Macro F1-Score    : {metrics_dict['macro_f1']:.4f}")
    lines.append(f"  Weighted Precision: {metrics_dict['weighted_precision']:.4f}")
    lines.append(f"  Weighted Recall   : {metrics_dict['weighted_recall']:.4f}")
    lines.append(f"  ROC-AUC (macro)   : {metrics_dict['auc_macro']:.4f}")

    lines.append("\n📋 PER-CLASS METRICS")
    lines.append("-"*40)
    lines.append(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} "
                 f"{'F1-Score':>10} {'Support':>10}")
    lines.append("  " + "-"*50)

    for cls in class_names:
        p  = report[cls]['precision']
        r  = report[cls]['recall']
        f1 = report[cls]['f1-score']
        s  = int(report[cls]['support'])
        full = config.CLASS_FULL_NAMES.get(cls, cls)
        lines.append(f"  {cls:<10} {p:>10.4f} {r:>10.4f} "
                     f"{f1:>10.4f} {s:>10}")

    lines.append("\n📁 OUTPUT FILES GENERATED")
    lines.append("-"*40)
    lines.append("  outputs/plots/confusion_matrix.png")
    lines.append("  outputs/plots/roc_curves.png")
    lines.append("  outputs/plots/training_history.png")
    lines.append("  outputs/plots/f1_per_class.png")
    lines.append("  outputs/reports/evaluation_report.txt")

    lines.append("\n" + "="*60)
    lines.append("  END OF REPORT")
    lines.append("="*60)

    # Write to file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))

    # Also print to terminal
    print('\n'.join(lines))
    print(f"\n✅ Evaluation report saved to: {save_path}")


# ============================================================
# QUICK TEST
# ============================================================
if __name__ == "__main__":
    print("Testing metrics.py...")
    print("✅ compute_all_metrics function loaded")
    print("✅ plot_confusion_matrix function loaded")
    print("✅ plot_roc_curves function loaded")
    print("✅ plot_training_history function loaded")
    print("✅ plot_f1_per_class function loaded")
    print("✅ save_evaluation_report function loaded")
    print("\n✅ metrics.py is ready!")
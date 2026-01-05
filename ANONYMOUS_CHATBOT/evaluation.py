"""
═══════════════════════════════════════════════════════════════════════════════
📊 EVALUATION - COMPREHENSIVE METRICS
═══════════════════════════════════════════════════════════════════════════════
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import json

from intent_classifier import IntentClassifier
from rag_config import EVAL_CONFIG, CONFUSION_MATRIX_PATH, CLASSIFICATION_REPORT_PATH

class Evaluator:
    def __init__(self):
        self.classifier = IntentClassifier()
    
    def plot_confusion_matrix(self, cm, labels, save_path=None):
        """Plot confusion matrix"""
        plt.figure(figsize=EVAL_CONFIG['confusion_matrix_figsize'])
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title('Confusion Matrix - Intent Classification')
        plt.ylabel('True Intent')
        plt.xlabel('Predicted Intent')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Confusion matrix saved: {save_path}")
        
        plt.close()
    
    def save_classification_report(self, report, save_path=None):
        """Save classification report"""
        if save_path is None:
            save_path = CLASSIFICATION_REPORT_PATH
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            for intent, metrics in report.items():
                if isinstance(metrics, dict) and 'precision' in metrics:
                    f.write(f"{intent}:\n")
                    f.write(f"  Precision: {metrics['precision']:.4f}\n")
                    f.write(f"  Recall: {metrics['recall']:.4f}\n")
                    f.write(f"  F1-Score: {metrics['f1-score']:.4f}\n")
                    f.write(f"  Support: {metrics['support']}\n\n")
        
        print(f"✅ Classification report saved: {save_path}")
    
    def evaluate_model(self, X_test, y_test, save_plots=True):
        """Complete evaluation"""
        print(f"\n{'='*80}")
        print("📊 COMPREHENSIVE EVALUATION")
        print('='*80)
        
        # Predictions
        y_pred = self.classifier.model.predict(X_test)
        
        # Metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred, labels=self.classifier.intent_labels)
        
        # Check thresholds
        accuracy = report['accuracy']
        f1_macro = report['macro avg']['f1-score']
        
        print(f"\n✅ Accuracy: {accuracy:.4f} (Threshold: {EVAL_CONFIG['min_accuracy']})")
        print(f"✅ F1-Score (Macro): {f1_macro:.4f} (Threshold: {EVAL_CONFIG['min_f1_score']})")
        
        if accuracy >= EVAL_CONFIG['min_accuracy']:
            print("✅ Accuracy threshold MET!")
        else:
            print("❌ Accuracy threshold NOT met")
        
        if f1_macro >= EVAL_CONFIG['min_f1_score']:
            print("✅ F1-Score threshold MET!")
        else:
            print("❌ F1-Score threshold NOT met")
        
        # Save
        if save_plots:
            self.plot_confusion_matrix(cm, self.classifier.intent_labels, CONFUSION_MATRIX_PATH)
            self.save_classification_report(report)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm
        }

if __name__ == '__main__':
    evaluator = Evaluator()
    # Load test data and evaluate
    print("Run training_pipeline.py first to generate test data")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import pointbiserialr
from scipy.stats import f_oneway
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import csv
import os

class MultiplePlotMaker:

    def __init__(self):
        pass
    
    def plot_multiple_distributions(self, df):
        # Plot histograms for each numeric feature
        numeric_columns = list(df.select_dtypes(include=['float64', 'int64']).columns)
        numeric_columns.pop(-1)
    
        df[numeric_columns].hist(bins=15, figsize=(15, 10), layout=(4, 3), edgecolor='black')
        plt.suptitle('Histograms of Numeric Features')
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # Adjust the top for the title
        plt.show()

    def plot_multiple_boxplots(self, df):
        # Plot boxplots for each numeric feature
        numeric_columns = list(df.select_dtypes(include=['float64', 'int64']).columns)
        numeric_columns.pop(-1)
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(numeric_columns, 1):
            plt.subplot(4, 3, i)
            sns.boxplot(y=df[col], color='skyblue')
            plt.title(f'Boxplot of {col}')
            plt.tight_layout()
        plt.suptitle('Boxplots of Numeric Features')
        plt.subplots_adjust(top=0.95)  # Adjust the top for the title
        plt.show()
        
    
    def plot_multiple_side_by_side_boxplots(self, df, cols_to_avoid = []):
        # Plot boxplots for each numeric feature
        column_names = list(df.select_dtypes(include=['float64', 'int64']).columns)
        target_variable_name = column_names.pop(-1)

        # Create boxplots to identify outliers
        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 10))

        # Flatten the axes array for easier iteration
        axes = axes.flatten()

        # Plot each boxplot
        for i, col in enumerate(column_names):
            if col not in cols_to_avoid:
                sns.boxplot(data=df, x=target_variable_name, y=col, ax=axes[i])
                axes[i].set_title(f'Side-by-side boxplot of {col}')

        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.show()

    def plot_multiple_overlapping_hist(self, df):
        column_names = list(df.columns)
        target_variable_name = column_names.pop(-1)  

        # Create subplots
        fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 10))

        # Flatten the axes array for easier iteration
        axes = axes.flatten()

        # Loop through each column to create overlapping histograms
        for i, col in enumerate(column_names):
            subset1 = df[df[target_variable_name] == 0][col]
            subset2 = df[df[target_variable_name] == 1][col]

            axes[i].hist(subset1, color="blue", label="0", density=True, alpha=0.5)
            axes[i].hist(subset2, color="red", label="1", density=True, alpha=0.5)

            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Density')
            axes[i].set_title(f'Overlapping Histogram of {col}')
            axes[i].legend()

        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.show()


class StatAnalysisTools:

    def __init__(self):
        pass

    def anova_test(self, df, target_col):
        features = df.select_dtypes(include=['float64', 'int64']).columns.drop(target_col)
        results = {}
        for feature in features:
            groups = [df[feature][df[target_col] == cls] for cls in df[target_col].unique()]
            f_val, p_val = f_oneway(*groups)
            results[feature] = p_val
        
        return results
    

    def mutual_info(self, df, target_col):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        mi = mutual_info_classif(X, y)
        mi_series = pd.Series(mi, index=X.columns)

        print( mi_series.sort_values(ascending=False))
        return

class ModelEvaluation:
    def __init__(self) -> None:
        pass
    
    def evaluate_classifier(self, X_test, y_test, trained_model):
        # Predict class labels
        y_pred = trained_model.predict(X_test)
        # Predict probabilities for ROC-AUC calculation
        y_pred_proba = trained_model.predict_proba(X_test)  # Multiclass probabilities

        # Calculate various metrics for multiclass
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')  # 'weighted' for multiclass imbalance
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')  # Specify 'ovr' for multiclass ROC-AUC
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Print the results
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
        print(f"ROC-AUC Score: {roc_auc:.3f}")
        print("Confusion Matrix:")
        print(conf_matrix)

        # Plot confusion matrix
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

        # Plot ROC curve for each class
        plt.figure(figsize=(8, 6))
        for i in range(len(trained_model.classes_)):
            # Compute ROC curve for each class
            fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba[:, i])  # One-vs-rest ROC curve
            # Calculate AUC for each class separately
            auc_score = roc_auc_score(y_test == i, y_pred_proba[:, i]) 
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_score:.3f})')

        plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title('Receiver Operating Characteristic (ROC) Curve for Each Class')
        plt.legend()
        plt.show()
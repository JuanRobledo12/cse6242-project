import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import pointbiserialr
from sklearn.model_selection import train_test_split
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
        
    
    def plot_multiple_side_by_side_boxplots(self, df):
        column_names = self.get_column_names()
        target_variable_name = column_names.pop(-1)  

        # Create boxplots to identify outliers
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))

        # Flatten the axes array for easier iteration
        axes = axes.flatten()

        # Plot each boxplot
        for i, col in enumerate(column_names):
            sns.boxplot(data=df, x=target_variable_name, y=col, ax=axes[i])
            axes[i].set_title(f'Side-by-side boxplot of {col}')

        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.show()

    def plot_multiple_overlapping_hist(self, df):
        column_names = self.get_column_names()
        target_variable_name = column_names.pop(-1)  

        # Create subplots
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))

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

    def perform_t_test(self, df):

        target_column_name = df.columns[-1] 

        for col in df.columns[:-1]:  # Exclude the target column
            group1 = df[df[target_column_name] == 0][col]
            group2 = df[df[target_column_name] == 1][col]
            t_stat, p_value = ttest_ind(group1, group2)
            print(f"{col}: t_stat = {t_stat},  p-value = {p_value}")
    
    def perform_correlation_binary_target(self, df):
        '''
        the point-biserial correlation (a special case of Pearson correlation for binary target variables) 
        to see how strongly each feature is associated with the target.
        '''

        for column in df.columns[:-1]:  # Exclude the target column
            correlation, p_value = pointbiserialr(df[column], df['Outcome'])
            print(f"{column}: correlation = {correlation}, p-value = {p_value}")

class ModelEvaluation:
    def __init__(self) -> None:
        pass
    
    def evaluate_classifier(self, X_test, y_test, trained_model, model_name, scaler_name, selected_features, csv_file_path):
        
        y_pred = trained_model.predict(X_test)
        y_pred_proba = trained_model.predict_proba(X_test)[:, 1]  # For ROC-AUC

        # Calculate various metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
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

        # Plot ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()

        # Append results to CSV
        file_exists = os.path.isfile(csv_file_path)
        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                # Write the header if the file doesn't exist
                writer.writerow(['model_name', 'scaler_name', 'selected_features', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'])
            
            # Write the data
            writer.writerow([model_name, scaler_name, selected_features, accuracy, precision, recall, f1, roc_auc])

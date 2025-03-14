import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calculate_dca(true_labels, predicted_probs, thresholds=np.arange(0, 1.05, 0.05)):

    net_benefit = []
    for threshold in thresholds:

        tp = np.sum((predicted_probs >= threshold) & (true_labels == 1))

        fp = np.sum((predicted_probs >= threshold) & (true_labels == 0))

        tn = np.sum((predicted_probs < threshold) & (true_labels == 0))

        fn = np.sum((predicted_probs < threshold) & (true_labels == 1))

        net_benefit.append((tp + tn - fp - fn) / len(true_labels) - (threshold * (fp + fn)))

    dca_data = pd.DataFrame({
        'Threshold': thresholds,
        'Net Benefit': net_benefit
    })
    
    return dca_data

def plot_dca(dca_data, result_path, model_name, timestamp):

    plt.figure(figsize=(10, 6))
    plt.plot(dca_data['Threshold'], dca_data['Net Benefit'], label=f'{model_name} (DCA)', color='blue', lw=2)
    plt.axhline(0, color='black', linestyle='--', lw=2)
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title(f'Decision Curve Analysis (DCA) - {model_name}')
    plt.legend(loc='upper left')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{result_path}/dca_curve_{model_name}_{timestamp}.png')
    plt.close()

def save_dca_values(dca_data, result_path, model_name, timestamp):

    dca_data.to_csv(f'{result_path}/dca_values_{model_name}.csv', index=False)

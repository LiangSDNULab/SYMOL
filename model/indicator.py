import numpy as np
from sklearn.metrics import (
    mutual_info_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    fowlkes_mallows_score,
    adjusted_rand_score,
    v_measure_score,
    f1_score,
    jaccard_score,
    completeness_score
)
from scipy.optimize import linear_sum_assignment


# 1. 读取标签
def read_labels_from_file(file_path):
    with open(file_path, 'r') as file:
        labels = [int(line.strip()) for line in file if line.strip().isdigit()]
    return np.array(labels)


# 2. 映射标签
def map_labels(true_labels, predicted_labels):
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    true_classes = np.unique(true_labels)
    predicted_classes = np.unique(predicted_labels)

    cost_matrix = np.zeros((len(true_classes), len(predicted_classes)))
    for i, true_class in enumerate(true_classes):
        for j, predicted_class in enumerate(predicted_classes):
            cost_matrix[i, j] = np.sum((true_labels == true_class) & (predicted_labels == predicted_class))

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    mapping = {predicted_classes[col]: true_classes[row] for row, col in zip(row_ind, col_ind)}
    mapped_predicted_labels = np.array([mapping[label] for label in predicted_labels])
    return mapped_predicted_labels


# 3. 计算指标
def evaluate_clustering(true_labels, predicted_labels):
    results = {
        "Mutual Information (MI)": mutual_info_score(true_labels, predicted_labels),
        "Normalized Mutual Information (NMI)": normalized_mutual_info_score(true_labels, predicted_labels),
        "Adjusted Mutual Information (AMI)": adjusted_mutual_info_score(true_labels, predicted_labels),
        "Fowlkes-Mallows Index (FMI)": fowlkes_mallows_score(true_labels, predicted_labels),
        "Adjusted Rand Index (ARI)": adjusted_rand_score(true_labels, predicted_labels),
        "V-Measure": v_measure_score(true_labels, predicted_labels),
        "F1-Score": f1_score(true_labels, predicted_labels, average='macro'),
        "Jaccard Similarity Coefficient (Jaccard)": jaccard_score(true_labels, predicted_labels, average='macro'),
        "Completeness (Compl.)": completeness_score(true_labels, predicted_labels)
    }
    for key, value in results.items():
        print(f"{key}: {value:.4f}")


# 4. 主程序
true_labels = read_labels_from_file('../Data/HLN/GT_labels.txt')
predicted_labels = read_labels_from_file('../results/HLN.txt')

mapped_predicted_labels = map_labels(true_labels, predicted_labels)

evaluate_clustering(true_labels, mapped_predicted_labels)

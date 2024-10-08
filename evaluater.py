import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np
import os


# Function to evaluate the model
def evaluate_model(model, test_dataloader, criterion, output_path):
    print("Evaluation startedd...")
    actual_labels = []
    predicted_labels = []
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            actual_labels.append(labels)
            predicted_labels.append(outputs)
            if(len(actual_labels)>100):
                break
    epoch_loss = running_loss / len(test_dataloader.dataset)

    print("Evaluation completed.")
    print(f'Test Loss: {epoch_loss:.4f}')

    # Create a directory in the output path to save the outputs
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Display the actual vs predicted values for each label separately as scatter plot
    scatter_plot = plot_scatter(actual_labels, predicted_labels)
    # Save the plots as images
    scatter_plot.savefig(os.path.join(output_path, 'scatter_plot.png')) 
    plt.show()

    roc_and_pr_plot = plot_ROC_and_PR(actual_labels, predicted_labels)
    # Save the plots as images
    roc_and_pr_plot.savefig(os.path.join(output_path, 'roc_and_pr_plot.png'))

    p_idx_loss, s_idx_loss = calculate_MSE_loss(actual_labels, predicted_labels)

    # Save the test loss as, p index MSE loss and S index MSE loss a text file
    # Save the ROC and PR plots as images
    # Save the actual and predicted values as a csv file
    
    with open(os.path.join(output_path, 'losses.txt'), 'w') as f:
        f.write(f'Test Loss: {epoch_loss:.4f}\n')
        f.write(f'P Index Loss: {p_idx_loss:.4f} seconds\n')
        f.write(f'S Index Loss: {s_idx_loss:.4f} seconds\n')
    with open(os.path.join(output_path, 'actual_predicted.csv'), 'w') as f:
        f.write('Actual P Index, Actual S Index, Actual P Confidence, Actual S Confidence, Predicted P Index, Predicted S Index, Predicted P Confidence, Predicted S Confidence\n')
        for i in range(len(actual_labels)):
            f.write(','.join([str(x) for x in actual_labels[i].numpy().flatten()]))
            f.write(',')
            f.write(','.join([str(x) for x in predicted_labels[i].numpy().flatten()]))
            f.write('\n')
    model.train()



#----------------------------------------------------- Helper Functions -----------------------------------------------------

def calculate_MSE_loss(actual_labels, predicted_labels):
    # Calculate MSE loss for indices, for the values where label is NOT [0,0,0,0]
    actual_labels = torch.cat(actual_labels)
    predicted_labels = torch.cat(predicted_labels)
    mse = nn.MSELoss()
    p_index_loss = mse(actual_labels[actual_labels[:,0] != 0][:,0], predicted_labels[actual_labels[:,0] != 0][:,0])
    s_index_loss = mse(actual_labels[actual_labels[:,1] != 0][:,1], predicted_labels[actual_labels[:,1] != 0][:,1])
    # MSE loss is the mean squaer error between the actual and predicted values in indices, now convert them to seconds, sample rate is 100Hz
    p_index_loss = p_index_loss * 0.01
    s_index_loss = s_index_loss * 0.01
    print(f'P Index Loss: {p_index_loss:.4f} seconds')
    print(f'S Index Loss: {s_index_loss:.4f} seconds')
    return p_index_loss, s_index_loss

def plot_ROC_and_PR(actual_labels, predicted_labels, p_index_tolerance_seconds=1, s_index_tolerance_seconds=1, p_confidence_threshold=0.4, s_confidence_threshold=0.4):
    # Concatenate the list of tensors into a single tensor
    actual_labels = torch.cat(actual_labels)
    predicted_labels = torch.cat(predicted_labels)
    
    p_index_tolerance_samples = p_index_tolerance_seconds * 100
    s_index_tolerance_samples = s_index_tolerance_seconds * 100

    ground_truth_condition_p_index = actual_labels[:,0] > 0
    # p index prediction condition = (p>0 and a=0 and abs(p-a)>tolerance) or (p>0 and abs(p-a)<=tolerance)
    prediction_condition_p_index = (((predicted_labels[:,0] > 0) & (actual_labels[:,0] == 0) & (torch.abs(actual_labels[:,0] - predicted_labels[:,0]) > p_index_tolerance_samples)) | 
                                    ((predicted_labels[:,0] > 0) & (torch.abs(actual_labels[:,0] - predicted_labels[:,0]) <= p_index_tolerance_samples))).int()

    ground_truth_condition_s_index = actual_labels[:,1] > 0
    # s index prediction condition = (s>0 and a=0 and abs(s-a)>tolerance) or (s>0 and abs(s-a)<= tolerance)
    prediction_condition_s_index = (((predicted_labels[:,1] > 0) & (actual_labels[:,1] == 0) & (torch.abs(actual_labels[:,1] - predicted_labels[:,1]) > s_index_tolerance_samples)) | 
                                    ((predicted_labels[:,1] > 0) & (torch.abs(actual_labels[:,1] - predicted_labels[:,1]) <= s_index_tolerance_samples))).int()

    # ROC index condition ROC(actual>0, ((predicted>0)and((abs(actual-predicted)<=tolerance) or (actual>0))))
    fpr_p_index, tpr_p_index, _ = roc_curve(ground_truth_condition_p_index.numpy(), prediction_condition_p_index.numpy())
    fpr_s_index, tpr_s_index, _ = roc_curve(ground_truth_condition_s_index.numpy(), prediction_condition_s_index.numpy())
    auc_p_index = auc(fpr_p_index, tpr_p_index)
    auc_s_index = auc(fpr_s_index, tpr_s_index)

    # ROC confidence condition ROC(actual>0, predicted>threshold)
    fpr_p_conf, tpr_p_conf, _ = roc_curve(actual_labels[:,2].numpy() > 0, predicted_labels[:,2].numpy() > p_confidence_threshold)
    fpr_s_conf, tpr_s_conf, _ = roc_curve(actual_labels[:,3].numpy() > 0, predicted_labels[:,3].numpy() > s_confidence_threshold)
    auc_p_conf = auc(fpr_p_conf, tpr_p_conf)
    auc_s_conf = auc(fpr_s_conf, tpr_s_conf)

    # For debugging purposes, print the first N actual and predicted labels, and the corresponding conditions
    N = 10
    print("Actual p index:", actual_labels[:N,0])
    print("Predicted p index:", predicted_labels[:N,0])
    print("Ground truth condition p index:", ground_truth_condition_p_index[:N])
    print("Prediction condition p index:", prediction_condition_p_index[:N])
    print("--------------------------------")
    print("Actual s index:", actual_labels[:N,1])
    print("Predicted s index:", predicted_labels[:N,1])
    print("Ground truth condition s index:", ground_truth_condition_s_index[:N])
    print("Prediction condition s index:", prediction_condition_s_index[:N])

    # Precision Recall index condition PR(actual>0, ((predicted>0)and((abs(actual-predicted)<=tolerance) or (actual>0))))
    precision_p_index, recall_p_index, _ = precision_recall_curve(ground_truth_condition_p_index.numpy(), prediction_condition_p_index.numpy())
    precision_s_index, recall_s_index, _ = precision_recall_curve(ground_truth_condition_s_index.numpy(), prediction_condition_s_index.numpy())
    avg_precision_p_index = average_precision_score(ground_truth_condition_p_index.numpy(), prediction_condition_p_index.numpy())
    avg_precision_s_index = average_precision_score(ground_truth_condition_s_index.numpy(), prediction_condition_s_index.numpy())

    # Precision Recall confidence condition PR(actual>0, predicted>threshold)
    precision_p_conf, recall_p_conf, _ = precision_recall_curve(actual_labels[:,2].numpy() > 0, predicted_labels[:,2].numpy() > p_confidence_threshold)
    precision_s_conf, recall_s_conf, _ = precision_recall_curve(actual_labels[:,3].numpy() > 0, predicted_labels[:,3].numpy() > s_confidence_threshold)
    avg_precision_p_conf = average_precision_score(actual_labels[:,2].numpy() > 0, predicted_labels[:,2].numpy() > p_confidence_threshold)
    avg_precision_s_conf = average_precision_score(actual_labels[:,3].numpy() > 0, predicted_labels[:,3].numpy() > s_confidence_threshold)

    # Plot ROC Curves
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.plot(fpr_p_index, tpr_p_index, label=f"P Index ROC (AUC = {auc_p_index:.2f})")
    plt.plot(fpr_s_index, tpr_s_index, label=f"S Index ROC (AUC = {auc_s_index:.2f})")
    plt.plot(fpr_p_conf, tpr_p_conf, label=f"P Confidence ROC (AUC = {auc_p_conf:.2f})")
    plt.plot(fpr_s_conf, tpr_s_conf, label=f"S Confidence ROC (AUC = {auc_s_conf:.2f})")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    # Plot Precision-Recall Curves
    plt.subplot(1, 2, 2)
    plt.plot(recall_p_index, precision_p_index, label=f"P Index PR (AP = {avg_precision_p_index:.2f})")
    plt.plot(recall_s_index, precision_s_index, label=f"S Index PR (AP = {avg_precision_s_index:.2f})")
    plt.plot(recall_p_conf, precision_p_conf, label=f"P Confidence PR (AP = {avg_precision_p_conf:.2f})")
    plt.plot(recall_s_conf, precision_s_conf, label=f"S Confidence PR (AP = {avg_precision_s_conf:.2f})")
    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()

    plt.tight_layout()
    return plt

def plot_ROC_and_PR_old(actual_labels, predicted_labels, p_index_tolerance_seconds=1, s_index_tolerance_seconds=1, p_confidence_threshold=0.4, s_confidence_threshold=0.4):
    # Concatenate the list of tensors into a single tensor
    actual_labels = torch.cat(actual_labels)
    predicted_labels = torch.cat(predicted_labels)
    
    p_index_tolerance_samples = p_index_tolerance_seconds * 100
    s_index_tolerance_samples = s_index_tolerance_seconds * 100

    ground_truth_condition_p_index = actual_labels[:,0] > 0
    prediction_condition_p_index = ((predicted_labels[:,0] > 0) & (((np.abs(actual_labels[:,0] - predicted_labels[:,0]) <= p_index_tolerance_samples) | (actual_labels[:,0] > 0))))

    ground_truth_condition_s_index = actual_labels[:,1] > 0
    prediction_condition_s_index = ((predicted_labels[:,1] > 0) & ((np.abs(actual_labels[:,1] - predicted_labels[:,1]) <= s_index_tolerance_samples) | (actual_labels[:,1] > 0)))

    # ROC index condition ROC(actual>0, ((predicted>0)and((abs(actual-predicted)<=tolerance) or (actual>0))))
    fpr_p_index, tpr_p_index, _ = roc_curve(ground_truth_condition_p_index, prediction_condition_p_index)
    fpr_s_index, tpr_s_index, _ = roc_curve(ground_truth_condition_s_index, prediction_condition_s_index)
    auc_p_index = auc(fpr_p_index, tpr_p_index)
    auc_s_index = auc(fpr_s_index, tpr_s_index)

    # ROC confidence condition ROC(actual>0, predicted>threshold)
    fpr_p_conf, tpr_p_conf, _ = roc_curve(actual_labels[:,2] > 0, predicted_labels[:,2] > p_confidence_threshold)
    fpr_s_conf, tpr_s_conf, _ = roc_curve(actual_labels[:,3] > 0, predicted_labels[:,3] > s_confidence_threshold)
    auc_p_conf = auc(fpr_p_conf, tpr_p_conf)
    auc_s_conf = auc(fpr_s_conf, tpr_s_conf)

    # For debugging purposes, print the first N actual and predicted labels, and the corresponding conditions
    N = 10
    print("Actual p index:", actual_labels[:N,0])
    print("Predicted p index:", predicted_labels[:N,0])
    print("Ground truth condition p index:", ground_truth_condition_p_index[:N])
    print("Prediction condition p index:", prediction_condition_p_index[:N])
    print("--------------------------------")
    print("Actual s index:", actual_labels[:N,1])
    print("Predicted s index:", predicted_labels[:N,1])
    print("Ground truth condition s index:", ground_truth_condition_s_index[:N])
    print("Prediction condition s index:", prediction_condition_s_index[:N])

    # Precision Recall index condition PR(actual>0, ((predicted>0)and((abs(actual-predicted)<=tolerance) or (actual>0))))
    precision_p_index, recall_p_index, _ = precision_recall_curve(ground_truth_condition_p_index, prediction_condition_p_index)
    precision_s_index, recall_s_index, _ = precision_recall_curve(ground_truth_condition_s_index, prediction_condition_s_index)
    avg_precision_p_index = average_precision_score(ground_truth_condition_p_index, prediction_condition_p_index)
    avg_precision_s_index = average_precision_score(ground_truth_condition_s_index, prediction_condition_s_index)

    # Precision Recall confidence condition PR(actual>0, predicted>threshold)
    precision_p_conf, recall_p_conf, _ = precision_recall_curve(actual_labels[:,2] > 0, predicted_labels[:,2] > p_confidence_threshold)
    precision_s_conf, recall_s_conf, _ = precision_recall_curve(actual_labels[:,3] > 0, predicted_labels[:,3] > s_confidence_threshold)
    avg_precision_p_conf = average_precision_score(actual_labels[:,2] > 0, predicted_labels[:,2] > p_confidence_threshold)
    avg_precision_s_conf = average_precision_score(actual_labels[:,3] > 0, predicted_labels[:,3] > s_confidence_threshold)

     # Plot ROC Curves
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.plot(fpr_p_index, tpr_p_index, label=f"P Index ROC (AUC = {auc_p_index:.2f})")
    plt.plot(fpr_s_index, tpr_s_index, label=f"S Index ROC (AUC = {auc_s_index:.2f})")
    plt.plot(fpr_p_conf, tpr_p_conf, label=f"P Confidence ROC (AUC = {auc_p_conf:.2f})")
    plt.plot(fpr_s_conf, tpr_s_conf, label=f"S Confidence ROC (AUC = {auc_s_conf:.2f})")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()

    # Plot Precision-Recall Curves
    plt.subplot(1, 2, 2)
    plt.plot(recall_p_index, precision_p_index, label=f"P Index PR (AP = {avg_precision_p_index:.2f})")
    plt.plot(recall_s_index, precision_s_index, label=f"S Index PR (AP = {avg_precision_s_index:.2f})")
    plt.plot(recall_p_conf, precision_p_conf, label=f"P Confidence PR (AP = {avg_precision_p_conf:.2f})")
    plt.plot(recall_s_conf, precision_s_conf, label=f"S Confidence PR (AP = {avg_precision_s_conf:.2f})")
    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()

    plt.tight_layout()
    return plt

def plot_scatter(actual_labels, predicted_labels):
    # Concatenate the list of tensors into a single tensor
    actual_labels = torch.cat(actual_labels)
    predicted_labels = torch.cat(predicted_labels)

    plt.figure(figsize=(14, 10))

    for i in range(2):
        for j in range(2):
            plt.subplot(2, 2, i*2+j+1)
            plt.scatter(actual_labels[:,i*2+j].numpy(), predicted_labels[:,i*2+j].numpy())
            plt.grid()
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            if(i == 0):
                if(j == 0):
                    plt.title('Actual vs Predicted for P index')
                else:
                    plt.title('Actual vs Predicted for S index')
            else:
                if(j == 0):
                    plt.title('Actual vs Predicted for P confidence')
                else:
                    plt.title('Actual vs Predicted for S confidence')
    return plt





    


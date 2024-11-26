import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
    def __init__(self, lambda_val):
        super(CustomLoss, self).__init__()
        self.lambda_val = lambda_val
        self.loss_e_lambda = 1

    def forward(self, outputs, labels):
        # We expect the outputs to be a tensor of shape (batch_size, 5)
        # The first two elements are the predicted P and S indices
        # The next three elements are the predicted P/S confidences: [1,0,0,0] for no P/S, [0,1,0,0] for P, [0,0,1,0] for S, [0,0,0,1] for both P/S

        p_predict, s_predict = outputs[:, 0], outputs[:, 1]
        confidence_pred = outputs[:, 2:]
        p_ground, s_ground = labels[:, 0], labels[:, 1]
        confidence_ground = labels[:, 2:]

        # Masks for different cases
        no_ps_mask = torch.all(confidence_ground == torch.tensor([1, 0, 0, 0]), dim=1)
        only_p_mask = torch.all(confidence_ground == torch.tensor([0, 1, 0, 0]), dim=1)
        only_s_mask = torch.all(confidence_ground == torch.tensor([0, 0, 1, 0]), dim=1)
        both_ps_mask = torch.all(confidence_ground == torch.tensor([0, 0, 0, 1]), dim=1)

        # Loss calculation is as follows: 
        # loss_e: cross entropy loss for the confidence
        # loss_p: MSE loss for P index
        # loss_s: MSE loss for S index
        # if there is no P/S in the ground truth --> total_loss = loss_e
        # if there is only P in the ground truth --> total_loss = lambda * loss_p + loss_e
        # if there are P/S in the ground truth --> total_loss = lambda * (loss_p + loss_s) + loss_e

        # Initialize total_loss
        total_loss = torch.zeros_like(p_predict)

        # Calculate the confidence loss, since it is always used
        loss_e = F.cross_entropy(confidence_pred, torch.argmax(confidence_ground, dim=1), reduction='none')

        # if there is no P/S in the ground truth, only use loss_e for total_loss
        if no_ps_mask.any():
            total_loss[no_ps_mask] = self.loss_e_lambda*loss_e[no_ps_mask]

        # if there is only P in the ground truth, additionally calculate loss_p
        if only_p_mask.any():
            loss_p = ((p_ground - p_predict) ** 2)
            total_loss[only_p_mask] = self.lambda_val * loss_p[only_p_mask] + self.loss_e_lambda*loss_e[only_p_mask]

        # if there is only S in the ground truth, additionally calculate loss_s
        if only_s_mask.any():
            loss_s = ((s_ground - s_predict) ** 2)
            total_loss[only_s_mask] = self.lambda_val * loss_s[only_s_mask] + self.loss_e_lambda*loss_e[only_s_mask]

        # if there are P/S in the ground truth, use lambda * (loss_p + loss_s) + loss_e        
        if both_ps_mask.any():
            loss_p = ((p_ground - p_predict) ** 2)
            loss_s = ((s_ground - s_predict) ** 2)
            total_loss[both_ps_mask] = self.lambda_val * (loss_p[both_ps_mask] + loss_s[both_ps_mask]) + self.loss_e_lambda*loss_e[both_ps_mask]

        return total_loss.mean()  # Return the average loss
        

    # def forward(self, outputs, labels):
    #     p_predict, s_predict = outputs[:, 0], outputs[:, 1]
    #     p_confidence, s_confidence = outputs[:, 2], outputs[:, 3]
    #     p_ground, s_ground = labels[:, 0], labels[:, 1]
    #     p_existence, s_existence = labels[:, 2], labels[:, 3]

    #     # Calculate individual loss components
    #     loss_p = p_existence * ((p_ground - p_predict) ** 2) # MSE for P index
    #     loss_s = s_existence * ((s_ground - s_predict) ** 2) # MSE for S index

    #     # Create the ground truth for classification, this is a multi-label classification problem
    #     # where the classes are [P, S] and the labels are [0, 1]
    #     class_ground_truth = torch.stack([p_existence, s_existence], dim=1)
    #     class_predict = torch.stack([p_confidence, s_confidence], dim=1)
    #     loss_class = F.cross_entropy(class_predict, class_ground_truth)

    #     # Combine losses
    #     total_loss = torch.mean(loss_p + loss_s + self.lambda_val * loss_class)
    #     return total_loss

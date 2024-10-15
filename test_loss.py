import torch   
import torch.nn.functional as F

def forward(outputs, labels):
        lambda_val = 0.5
        # We expect the outputs to be a tensor of shape (batch_size, 6)
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
        print("Confidence pred: ", confidence_pred)
        print("Confidence ground: ", confidence_ground)
        loss_e = F.cross_entropy(confidence_pred, torch.argmax(confidence_ground, dim=1), reduction='none')
        print("Loss e: ", loss_e)

        # if there is no P/S in the ground truth, only use loss_e for total_loss
        if no_ps_mask.any():
            total_loss[no_ps_mask] += loss_e[no_ps_mask]
            print("Total loss in case of no_ps_mask: ", total_loss)

        # if there is only P in the ground truth, additionally calculate loss_p
        if only_p_mask.any():
            loss_p = ((p_ground - p_predict) ** 2)
            total_loss[only_p_mask] += lambda_val * loss_p[only_p_mask] + loss_e[only_p_mask]
            print("Total loss in case of only_p_mask: ", total_loss)

        # if there is only S in the ground truth, additionally calculate loss_s
        if only_s_mask.any():
            loss_s = ((s_ground - s_predict) ** 2)
            total_loss[only_s_mask] += lambda_val * loss_s[only_s_mask] + loss_e[only_s_mask]
            print("Total loss in case of only_s_mask: ", total_loss)

        # if there are P/S in the ground truth, use lambda * (loss_p + loss_s) + loss_e        
        if both_ps_mask.any():
            loss_p = ((p_ground - p_predict) ** 2)
            loss_s = ((s_ground - s_predict) ** 2)
            total_loss[both_ps_mask] += lambda_val * (loss_p[both_ps_mask] + loss_s[both_ps_mask]) + loss_e[both_ps_mask]
            print("Total loss in case of both_ps_mask: ", total_loss)

        return total_loss.mean()  # Return the average loss


# Test the loss function
outputs = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
labels = torch.tensor([[0.1, 0.2, 0, 0, 1, 0], [0.2, 0.3, 0, 1, 0, 0], [0.3, 0.4, 0, 0, 0, 1]])
loss = forward(outputs, labels)
print(loss)
import numpy as np
import torch


def tensor_to_variable(x):
    tensor = x.float()
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def variable_to_numpy(x):
    tensor = x.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()


def print_preds(y_label_list, y_pred_list, test_or_tr):
    print()
    print("{} Set Predictions: ".format(test_or_tr))
    for true_value, pred_value in zip(y_label_list, y_pred_list):
        print("True:{}, Predicted: {}".format(true_value, pred_value))


def mse(Y_prime, Y):
    return np.mean((Y_prime - Y) ** 2)


def macro_avg_err(Y_prime, Y):
    if isinstance(Y_prime, np.ndarray):
        denom = np.sum(np.abs(Y))
        return np.sum(np.abs(Y - Y_prime)) / denom if denom != 0 else 0.0
    denom = torch.sum(torch.abs(Y))
    return torch.sum(torch.abs(Y - Y_prime)) / denom.clamp(min=1e-12)

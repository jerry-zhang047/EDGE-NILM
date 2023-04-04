import torch
from torch.quantization import quantize_dynamic
from model import CNN

if __name__ == '__main__':
    model = CNN(200)
    model.load_state_dict(torch.load("./run/washing machine_CNN_pruned_best_state_dict.pth"))
    model.eval()

    quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    torch.save(quantized_model.state_dict(), "./run/washing machine_CNN_pruned_quantized_best_state_dict.pth")
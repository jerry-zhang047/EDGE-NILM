import os
import torch
from tinynn.converter import TFLiteConverter
from model import CNN

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

def main_worker():
    model = CNN(200)
    model.load_state_dict(torch.load("./tflite_converter/washing machine_CNN_pruned_best_state_dict.pth"))

    dummy_input = torch.randn(256, 1, 200)

    output_path = os.path.join(CURRENT_PATH, 'out', 'WM_LIGHT.tflite')

    converter = TFLiteConverter(model, dummy_input, output_path)
    converter.convert()

if __name__ == '__main__':
    main_worker()
# EDGE-NILM

- **Introduction**

**EGDE-NILM** is my simple `final year project`. Non-intrusive Load Monitoring (NILM) estimates the appliance-level power consumption using just the whole-home power meter readings. The state-of-the-art method is the deep learning algorithm but it consumes large computation resources and memory storage. **EDGE-NILM** aims to design a lightweight AI model and deploy it on a microcontroller via STM32CubeMX x-cube-ai package.

---    

- **Environment Setup**  

    - If you use `conda`, we could run `conda create -f environment.yml` to install dependencies.
    - Then paste the entire `\nilmtk` folder under the directory of `nilmtk-env` environment, which is located at `\Anaconda3\envs\nilmtk-env\Lib\site-packages`.
    - x-cube-ai package does not support `PyTorch` models, so it is neccessary to convert into `TensorFlow Lite` models. [TinyNeuralNetwork](https://github.com/alibaba/TinyNeuralNetwork) is an easy-to-use deep learning framework conversion tool from PyTorch to TFLite. Follow the installation instructions to install the package in `nilmtk-env`.

---

- **References**

    - [NILMTK v0.4](https://github.com/nilmtk/nilmtk) provides a useful `API` class for conducting NILM experiments
    - For denoising autoencoder, please refer to [Kelly J, Knottenbelt W. Neural nilm: Deep neural networks applied to energy disaggregation](https://arxiv.org/abs/1507.06594)
    - For seq2seq and seq2point algorithms, please refer to [Zhang C, Zhong M, Wang Z, et al. Sequence-to-point learning with neural networks for non-intrusive load monitoring](https://arxiv.org/abs/1612.09106)
    - For lightweight NILM model prototype, please refer to [Luan, Zhang, Liu, Zhao, Yu: Leveraging sequence-to-sequence learning for online non-intrusive load monitoring in edge device](https://www.sciencedirect.com/science/article/pii/S0142061522009061)
    - The disaggregation code is based on [NeuralNILM_Pytorch](https://github.com/Ming-er/NeuralNILM_Pytorch)
---
- **Tutorials**

    - Refer to `\nilmtk\disaggregate` where the algorithms are placed.
    - Refer to `\example\launch_experiment.ipynb` where a demo experiment is conducted.
    - Refer to `\utils\pruner` where you can perform model pruning.
    - Refer to `\utils\quantization` where you can find quantization tools.
    - Refer to `\utils\tflite_converter` where you can convert your PyTorch model into TensorFlow Lite format. 
    - Refer to the video of [Getting Start with STM32 X-CUBE-AI](https://www.youtube.com/watch?v=crJcDqIUbP4&t=654s), where you can learn how to deploy an AI model on your STM32 microcontroller.
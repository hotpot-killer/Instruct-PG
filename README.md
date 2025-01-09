<h1 align="center">Instruct-PG: Enhancing Image Editing with Semantic and Preference Alignment</h1>
Instruct-PG - Official Implementation


## Introduction 
This repo, named **Instruct-PG**, contains the official PyTorch implementation of our paper Instruct-PG: Enhancing Image Editing with Semantic and Preference Alignment.
We are actively updating and improving this repository. If you find any bugs or have suggestions, welcome to raise issues or submit pull requests (PR).

## Getting Started 🏁
### 1. Clone the code and prepare the environment 

```bash
git clone https://github.com/yourusername/Instruct-PG.git
cd Instruct-PG

# create env using conda
conda create -n InstructPG python=3.10
conda activate InstructPG
pip install -r requirements.txt
```
# Nvidia users should install stable pytorch using this command:
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
```
### 2. Download pretrained weights

#### 2.1 Download Stable Diffusion v1.5 weights
```bash
# !pip install -U "huggingface_hub[cli]"
huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5
```
#### 2.2 Download ImageFLow weights
You can download our preference model from [here](#).
### 3. Train and Inference
```
python main.py
```

### 4. DataSets
You can download our Preference Dataset from [here](#).
You can also download our Image Editing Instruction Dataset from [here](#).

## Contact

For any queries or collaboration opportunities, please contact us at [wuzhenhua992@gmail.com](mailto:wuzhenhua992@gmail.com).

---

Feel free to explore the code, raise issues, or submit pull requests to enhance the framework further. Happy editing!

### Open Source Plan
We are excited to announce that Instruct-PG will be fully open-sourced in the near future! We aim to promote innovation and development within the community by making the source code freely available, thereby advancing image editing technology together.
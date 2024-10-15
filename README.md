## Dataset:
After downloading the dataset, follow these steps:

1. **Extract the Dataset:**
   - Unzip or extract the downloaded dataset to a desired location on your computer.

2. **Update Dataset Paths:**
   - You need to modify the dataset paths in the respective dataloader scripts for both TensorFlow and PyTorch.

### TensorFlow:
- In the TensorFlow dataloader file, locate **lines 55, 56, and 57**. Replace these lines with the path to the `train`, `validation`, and `test` folders on your computer.

### PyTorch:
- In the PyTorch dataloader file, go to line 7 and provide the path to the dataset folder.


## Usage
This repository contains the `main_mdnet` script, which is designed for both training and evaluating machine learning models. The script accepts three arguments: `--framework`, `--model`, and `--eval`. Below is a detailed guide on how to use the script. **Note:** The results presented in the paper are based solely on the TensorFlow model. The PyTorch implementation has not been tested as of the most recent update.

### 1. Framework Selection (`--framework`):
- `pytorch`: Choose this to run the script using the PyTorch framework.
- `tensorflow`: Choose this to run the script using the TensorFlow framework.
- The Pytorch and TensorFlow models should be run in separate Conda environments.

### 2. Model Selection (`--model`):
- `69`: Use model 69 for training.
- `117`: Use model 117 for training.

### 3. Evaluation Mode (`--eval`):
- `69` or `117`: If the `--eval` argument is specified with either `69` or `117`, the script will run in evaluation mode for the respective model.
- Only needed for TensorFlow.

---

## TensorFlow Usage:
To train a TensorFlow model, use the following command structure:

```bash
python main_mdnet.py --framework tensorflow --model [69|117]
```
If you also want to evaluate the TensorFlow model, provide the --eval argument:
```bash
python main_mdnet.py --framework tensorflow --model [69|117] --eval [69|117]
```

---

## Pytorch Usage:
For PyTorch, training and evaluation are combined. The script will automatically handle both processes when run. Here's how to execute the PyTorch version:

```bash
python main_mdnet.py --framework pytorch --model [69|117]
```
There is no need for an explicit --eval argument when using PyTorch.


## Example Commands:
- Train MeDiANet-69 with TensorFlow:
```bash
python main_mdnet.py --framework tensorflow --model 69
```

- Train and evaluate MeDiANet-117 with TensorFlow: 
```bash
python main_mdnet.py --framework tensorflow --model 117 --eval 117
```

- Train and evaluate MeDiANet-69 with PyTorch:
```bash
python main_mdnet.py --framework pytorch --model 69
```

# WHMM
A Hidden Markov Model (HMM) implementation that predicts the next state in a given grid based on historical state sequences is named 'Walter - HMM (WHMM)'.

WHMM is applicable for Semi-Persistent Scheduling (SPS) based on the multiple steps of  regular historical data.

## Notations

- The slot number in a selection window for Vulnerable Road User (VRU): $\tau$.


- The independent subchannels number in the resource pool: $N$.


- The number of historical data: $t$.


- X-coordinates of the TB usage in the $t$-th selection window corresponding to the subchannel number: $x_t\in[1,2,\cdots,N]$.


- Y-coordinates of the TB usage in the $t$-th selection window corresponding to the slot number: $y_t\in[1,2,\cdots,\tau]$.


- The position of the TB usage in the $t$-th selection window: $[x_{t}, y_{t}]$.


## Overview
This project trains an GRU-based model to predict $[x_{t+1}, y_{t+1}]$ given the rule of SPS algorithm and the historical data $[x_{1}, y_{1}],[x_{2}, y_{2}],\cdots,[x_{t}, y_{t}]$. 

The model treats the task as a $(N\times\tau)$-class classification problem, and each class corresponds to one TB in the selection window. 

## Dataset

- Input: Sequences length of $t$ steps, and each step containing two-dimensional data: $[x_{1}, y_{1}],[x_{2}, y_{2}],\cdots,[x_{t}, y_{t}]$.
- Output: $C_{t+1}$ for the $(t+1)$-th selection window: $C_{t+1}=(y_{t+1}-1)\times N+x_{t+1}-1$, where $C_{t+1}\in[0,1,2,\cdots,(N\times\tau-1)]$.
- Format: The input dataset is a CSV file, where odd rows contain X-coordinates ($x_1, x_2,\cdots,x_t$ values corresponding to the subchannel number) and even rows contain Y-coordinates ($y_1,y_2,\cdots,y_t$ values corresponding to the  slot number).
  

Example data structure:
| Row Numbuer | Values           |
| --- | ---------------- |
| 1 (X1)  | 1, 1, 3, ...      |
| 2 (Y1)  | 11, 11, 16, ...   |

## Installation

### 1. Clone the repository:
git clone https://github.com/yyymk/WLSTM-WGRU-WHMM.git

### 2. Install dependencies:
pip install -r requirements.txt

numpy
pandas
scikit-learn
matplotlib

## Usage
1. Prepare your dataset in CSV format similar to RP_power_data.csv (included in the repo).

2. Run the training script:
   python3 WHMM_train.py

3. After training, the script will output $C_{t+1}$ for the $(t+1)$-th selection window. The position $[x_{t+1}, y_{t+1}]$ in the $(t+1)$-th selection window can be calculated by: $C_{t+1}=(y_{t+1}-1)\times N+x_{t+1}-1$, $x_{t+1}\in[1,2,\cdots,N]$, and $y_{t+1}\in[1,2,\cdots,\tau]$.
   
4. By comparing the actual position with the predicted one, the accuracy can be obtained.

## Command-line Arguments	
| Argument      | Default | Description                                              |
| ------------- | ------- | -------------------------------------------------------- |
| `--data`      | None    | Path to your dataset CSV file                            |
| `--seq_len`   | 50      | Length of the sequence                                   |
| `--grid_x`    | 4       | Number of columns in the grid                            |
| `--grid_y`    | 50      | Number of rows in the grid                               |
| `--alpha`     | 1.0     | Laplace smoothing parameter for transition probabilities |
| `--test_size` | 0.2     | Proportion of the data to use for testing                |
| `--k`         | 3       | Top-k accuracy evaluation                                |


## Model Architecture
The model is a HMM  for classification:

- Input: $t$ two-dimensional historical data $[x_{1}, y_{1}],[x_{2}, y_{2}],\cdots,[x_{t}, y_{t}]$.
- Initial Distribution (𝜋): Estimated using the frequency of the first state in each sequence from the training data.
- Transition Matrix (𝐴): A matrix representing the probability of transitioning from one state to another, estimated using the frequencies of consecutive states in the training data, with Laplace smoothing applied to avoid zero probabilities.
- Output: $(N\times\tau)$ classes for the $(t+1)$-th selection window.
- Loss: CrossEntropyLoss for multi-class classification.

## Trained Model（WHMM）
- Dataset Size: The model was trained on $5940$ sequences of trajectory data.
- Data Split: The dataset was split into training and testing sets with a ratio of **$80%$ training and $20%$ testing**.
- Training Parameters:

    Transition Matrix Smoothing: $\alpha = 1.0$ (Laplace smoothing parameter for transition probabilities)

    Top-k: $3$;

    Optimizer: The model doesn't require an optimizer in the typical sense, as it is a statistical model based on state transitions. However, hyperparameters like smoothing ($\alpha$) can be tuned;

    Loss Function: CrossEntropyLoss for multi-class classification.

- In our demo with default parameter settings ($\tau=50, N=4$, and $t=50$), the model achieved an accuracy of **67.23%** on the test set after training.

## License
This project is licensed under the MIT License.

# FedProx for MNIST: A Federated Learning Implementation in PyTorch

This project implements the **FedProx** algorithm for federated learning on the MNIST dataset, focusing on non-IID data scenarios.  
This is a beginner-friendly project — it's also the **first full project I've written myself**, so I’ve included very detailed comments in the code.  
I hope this can be helpful to readers and users who are learning federated learning just like I did. ><  

## Installation

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure MNIST dataset availability**:

   - The MNIST dataset is handled by `MnistDataloader.py` and stored in the `mnist/` directory.

## Usage

Run the FedProx server:

```bash
python server.py
```

This executes federated learning with 10 clients, 30 epochs, and a 60% client participation rate.

To modify hyperparameters, edit:

- `client.py`: Adjust `learning_rate` (default: 0.01), `mu` (default: 0.1), `local_epoch` (default: 5).
- `server.py`: Adjust `cfraction` (default: 0.6), `epoch` (default: 30).

## Project Structure

```
FedProx/
├── __pycache__/            # Python bytecode cache directory, automatically generated during execution
├── mnist/                  # Directory containing MNIST dataset files
├── client.py               # Client-side logic: model definition, local training, proximal term
├── server.py               # Server-side logic: client selection, parameter aggregation
├── data_prework.py         # Data preprocessing and non-IID data partitioning
├── MnistDataloader.py      # Utility for loading and preprocessing MNIST dataset
├── requirements.txt        # Dependency list
├── README.md               # This file
```

## Contact

- Author: Zheng-Xin,Xi
- Email: schumi7611@gmail.com
- GitHub: EasternMichael

## Acknowledgments

- Inspired by the FedProx paper: *Federated Optimization in Heterogeneous Networks* (Li et al., 2020).
- Special thanks to my kind senior **Jon** for his patient guidance and support throughout the project.
- Neural network model adapted from this Medium article.

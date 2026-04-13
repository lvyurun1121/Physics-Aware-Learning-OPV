# Physics-Aware-Learning-OPV
Core implementation and sample dataset for "Physics-aware learning of donor-acceptor pair interactions for high-efficiency organic photovoltaics" (Under Revision/Submitted, 2026).
Physics-Aware Learning of Donor-Acceptor Pair Interactions for OPV
This repository provides the core model architecture and a sample dataset for the research paper:
"Physics-aware learning of donor-acceptor pair interactions for high-efficiency organic photovoltaics"
(Currently under revision/submitted).
🌟 Overview
This project introduces a physics-aware Graph Neural Network (GNN) to predict the performance of organic photovoltaic (OPV) donor-acceptor (D-A) pairs. The implementation focuses on modeling molecular interactions through a specialized Pair Interaction Block.
📁 Files in this Repository
model.py: The primary Python script containing the core PyTorch modules:
GNNEncoder: GINEConv backbone for molecular graph encoding.
PairInteractionBlock: Explicit modeling of ∣D−A∣and D⊙Ainteraction terms.
PCEPredictor: The complete model for multi-task prediction of Jsc, Voc, FF, and PCE.
multi_task_loss: The uncertainty-weighted loss function.
sample_data.csv: A representative subset (~300 samples) of our curated dataset. It includes SMILES strings and experimental PCE values to demonstrate the data format required by the model.
🛠️ Key Dependencies
PyTorch & PyTorch Geometric (For the GNN architecture)
RDKit (For molecular processing)
Scikit-learn (For data preprocessing)
📖 Citation
If you reference our methodology or use this code, please cite our work as:
Lyu, Y. et al., Physics-aware learning of donor-acceptor pair interactions for high-efficiency organic photovoltaics. (Under Revision/Submitted, 2026).

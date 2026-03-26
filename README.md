# Effi-TAD
## Introduction
This repository contains the official code for the paper "Effi-TAD: Efficient Temporal Action Detection via Temporal Interaction and Boundary-Aware Modeling".

This paper proposes a novel temporal action detection framework with two core components, namely a Temporal Feature Interaction Module (TFIM) and a Tri-Branch Boundary Aware Head (TriBE Head). The Temporal Feature Interaction Module is applied after the backbone to enhance inter-snippet temporal relations, enabling adaptive information propagation across long sequences while preserving computational efficiency and avoiding interference with backbone feature extraction. The Tri-Branch Boundary Aware Head further improves boundary sensitive detection by decoupling the prediction of start boundary, end boundary and center offset into three specialized branches.

![Model Framework](\docs\figures\overview.png)

## 🛠️ Preparation

Please prepare the environment, data, and models as described in [preparation.md](docs/en/preparation.md).

## Contact
If you have any questions regarding the code or paper, please contact us via the email on the paper homepage.
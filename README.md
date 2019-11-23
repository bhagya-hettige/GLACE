# GLACE
Gaussian Embedding of Large-scale Attributed Graphs

Split the dataset
-----------------
python split_data.py --name=<name> --p_val=<p_val> --p_test=<p_test>

Train with LACE/GLACE
---------------------
python train.py <name> <lace|glace> --proximity=<first-order|second-order> --is_all=<all_edges> etc.


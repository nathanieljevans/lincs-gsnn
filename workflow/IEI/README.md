# Input Edge Inference (IEI)

This workflow is intended to evaluate the ability of this project to infer Drug-target interactions by formulating them as input edge inference. 

To do this, we first create a prior knowledge graph while holding out several known DTIs. Then we train our model using trajectories from LINCS-TRAJ. 
Lastly, we run the input edge inference algorithm. 

# TODO ; in dev 
- create a input_edge_inference script 
- add `hold_out_input_edges` to make_bio_network 
- create evaluation script 
- update the snakemake and config files to adapt appropriately. 

how many edges to hold out? should there always be one DTI? 


# Overview and Objective
 To make a semantic content based medical image retrieval system for Retinal Fundus images. This is a POC on diabetic retinopathy grading task using the DeepDRiD dataset. Basically, we wish to create a robust Approximate Nearest Neighbor Search Data-structure, comprised of feature-vectors which are representative of individual images in the train set. These vectors can be obtained anyhow, with the best method being using neural networks. Basic CNN architecture is used in this repo. For actually creating a data-structre, HNSWGs (Hierarchial Navigable Small World Graphs) are used. Other structures such as VP-Trees, Ball Trees etc can also be explored.

# Current Used Method
 1. selected relevant examples from DeepDriD Regular Fundus images, based on overall quality and clarity.
 2. divided these into "train" and "validation" sets. Here, features extracted from the train images are used to create a vector
    db. The one used currently is Pinecone (which internally creates an HNSWG - Hierarchial Navigable Small World Graph). Validation images are then used for ANN (Approximate Nearest Neighbor) queries. 
 3. Currently, the system performance is measured using confusion matrix, top-1 and top-5 accuracy. 
 4. Data selection part is available in <u>rough.py</u>
 5. The DeepDRiD folder in this repo has the selected data incices. Actual images can be found from link in appendix.

# Issues
 1. current top-1 accuracy is only ~0.51 (<u>pinecone-implementation.ipynb</u>). This could be due to multiple reasons, such as dataset-
  -imbalance. Majority images are from class 0, with very few from class 4. It could also be due to the fact that the model being used was trained with the EyePACS dataset, and needs to be fine-tuned. It could also be an issue with the model itself. Lastly, it could be an issue with the graph formed by the vector database. This is hard to verify since we do not have control over its creation. 
 2. After applying image augmentations, accuracy actually reduced to 0.43 (<u>pinecone_DRiD_2.ipynb</u>).

# Changes Considered
 1. use an open-source HNSWG implementation, like FAISS (Facebook AI Similarity Search).
 2. Fine tune the model, or re-train with different architecture.
 3. Use other metrics for model evaluation, so as to make results more relevant in the research community, i.e. Rank-aware evaluation metrics (https://medium.com/swlh/rank-aware-recsys-evaluation-metrics-5191bba16832)

# Appendix
 Currently used model and weights: https://github.com/YijinHuang/Lesion-based-Contrastive-Learning 
 paper for above: https://paperswithcode.com/paper/lesion-based-contrastive-learning-for 

 Dataset: https://drive.google.com/drive/folders/1dAx7CPUpoTFzWeA_31OZmDaW9q1CJwB8?usp=sharing 

 Other implimentations: https://paperswithcode.com/task/diabetic-retinopathy-grading 
 Another possibly great approach, especially for discriminative feature extraction: https://paperswithcode.com/paper/robust-collaborative-learning-of-patch-level#code 



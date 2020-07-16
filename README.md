# Neural Collaborative Filtering

Pythorch Version of `Neural Collaborative Filtering` at WWW'17 

![model](https://dnddnjs.github.io/assets/img/Untitled-a3e99bbb-bc1c-4f0a-92e9-b15746d9e4d8.png)

**[Paper](https://dl.acm.org/doi/10.1145/3038912.3052569)**

**[Official_Code(Keras)](https://github.com/hexiangnan/neural_collaborative_filtering)**

**[Author: Dr. Xiangnan He](http://www.comp.nus.edu.sg/~xiangnan/)**


## Keypoints
* The problem that the thesis intends to solve is to recommend the item to the user based on implicit feedback.
* Applying deep learning to user-item interaction in matrix factorization
* Using a network structure that takes advantage of both dot-product (GMF) and MLP
* Use binary cross-entropy rather than MSE as loss function
* Use point-wise loss + negative sampling

## Results
![Screenshot](https://user-images.githubusercontent.com/47301926/87726445-f934e980-c7f9-11ea-9daa-07d48bc0d7ec.png)

Best epoch 005: HR = 0.677, NDCG = 0.399

## Requirements
`python==3.7.7`
`pandas==1.0.3`
`numpy==1.18.1`
`torch==1.4.0`

## Refrence
* [Evaluation Part](https://github.com/guoyang9/NCF)
* [Data Part](https://github.com/microsoft/recommenders/blob/master/reco_utils/recommender/ncf/dataset.py)

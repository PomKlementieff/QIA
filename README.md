# Quantum Inception Architecture (QIA) via TF 2.8
### Update (2023/09/21)
- We are enhancing the quality of the code to improve its visibility. This will make it easier to understand how QIA works in the future.


- While the algorithm has been implemented, it requires optimization through hyperparameter tuning.


# Run Experiments
```c
python qcnn.py --batch_size 32 --dataset fmnist --epochs 30
```

# Experimental Results
- We evaluated the performance of QIA against three models: Fully Connected (FC), Convolutional Neural Network (CNN), and Quantum Convolutional Neural Network (QCNN).


- The algorithms are implemented as individual functions within the 'nets' folder.


- QIA consistently demonstrated superior classification results compared to QCNN. The use of the quantum inception module was directly linked to improved training performance.


- In most cases, when compared to classic FC and CNN, QIA exhibited significantly better performance.

## Performance Comparison of Models Trained on the Fashion MNIST Dataset: Error Rate, Loss, F1-score, and Duration

|Model|Error Rate|Loss|F1-score|Duration|
|:----------|:----------:|:----------:|:----------:|:----------:|
|FC|15.51%|0.4426|84%|**0:00:12**|
|CNN|13.22%|0.3733|87%|0:00:34|
|QCNN|16.61%|0.4681|83%|0:22:36|
|QIA(naive version)|**12.07%**|**0.3478**|**88%**|0:05:33|
|QIA(with dimension reductions)|13.46%|0.3859|87%|0:27:58|

# Plan to do
- Check all the algorithms.


- Conduct experiments.

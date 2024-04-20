
### Content implementation
Siamese network is implemented in this warehouse for training character evolution. The backbone network used by the warehouse are VGG16,resnets 50,and alexnet.

### Environment configuration
tensorflow-gpu==2.2.0
keras=2.1.5
### Matters needing attention
**First of all, I conducted experiments with omniglot data set, and trained my data set to use two different formats, so I need to pay attention to the formatting.
**Then, three experiments were compared, and the codes were named respectively

### Prediction steps
#### 1、Use pre-trained weights
Run：predict.py，
Follow the steps to complete the prediction
#### 2、Use your own training weights
a、Follow the training steps.
b、In the siamese.py file, modify model_path to correspond to the trained file in the following sections；**Model_path corresponds to the weight file under the logs folder**。
```python
_defaults = {
    "model_path": 'model_data/Omniglot_vgg.h5',
    "input_shape" : (105, 105, 3),
}
```
c、 Run predict.py

#### 1、
The our dataset stores data in two levels：
```python
- image_background
	- Ai_01
			- 0000_01.png
			- 0000_02.png
			- ……
	- Bing_03
	- ……
```
The training steps are:
a、Run train.py to start training
#### 2、：
a、Put the dataset in the above format and under the dataset folder in the root directory.
b、Then set train_own_data in train.py to True.
c、 Run train.py to start training.



## 금융 데이터를 딥러닝으로 모델링 하기 힘든 이유 

### 문제점 1. 시계열 feature 자체의 노이즈
일반적으로 주가 모델링은 기하브라운모형(GBM)을 가정하지만, 근본적으로 AR(1)과 크게 다르지 않으므로 편의를 위해 AR(1) 모형에서 시작해 조금만 변

$$ y_{t+1} = y_t + x_t + \epsilon_t $$
$$ 다음 주가 = 현재 주가 + 정보 + 노이즈 $$

이 때, 일반적으로 금융 데이터는 $\epsilon_t > x_t$가 되어 정보보다 노이즈가 더 많은 형태이다.
그렇기 때문에 결국에는 다음과 같이 모델링이 되어버린다.

$$ y_{t+1} = y_t + \epsilon_t $$


즉, 주어진 정보를 다음의 주가로 반영을 하지 못하고 현재의 주가에 대한 그대로의 반영이 되어 예측을 할 수가 없어진다.

#### 해결책 1. Time-series denoising

* Moving Aervage
> 단순히 오른쪽으로 lagging 될뿐...

* Bilateral Fiter
> lagging은 없어졌으나 결국에는 충분치 않음...

* Auto-Encoder
> feature를 추출하기 전에 AE활용하여 CNN Stacked AutoEncoder를 활용하여 denoising module로서 활용

input data -> AE -> feature extraction -> output data
 

### 문제점 2. 시계열 feature 종류 대비 짧은 시계열 길이

* 기존의 feature를 잘 활용하자
> 기존의 feature에 attention을 활용하여 weighting 주자

* 새로운 데이터를 만들어 보자
> GAN? 기존의 


### 문제점 3. 문제점 1과 2로 인한 overfitting
* 딥러닝의 경우 모델의 initialization을 어떻게 하느냐에 따라 도달하는 점이 다름
> 여러 경우의 initailization을 활용

> AE를 unsupervised learning의 pre-trained net의 weight를 initail weight로 활용

* Bayesian Inference를 활용하여 관찰되지 못한 부분에 대해서 uncertainty를 추정

> Monte Carlo Dropout, Gal, Y., 2016

> Monte Carlo Batch Normalization, Teye, M., 2018

> Deep learning Regression + Gaussian Process Regression 
>> 마지막 단의 노드들을 이용하여 GPR을 활용하여 uncertainty를 추정 


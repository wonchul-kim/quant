# 금융 데이터를 딥러닝으로 모델링 하기 힘든 이유 

## 문제점 1. 시계열 feature 자체의 노이즈

일반적으로 주가 모델링은 기하브라운모형(GBM)을 가정하지만, 근본적으로 AR(1)과 크게 다르지 않으므로 편의를 위해 AR(1) 모형에서 시작해 조금만 변한다.

$$ y_{t+1} = y_t + x_t + \epsilon_t $$

$$ 다음 주가 = 현재 주가 + 정보 + 노이즈 $$

이 때, 일반적으로 금융 데이터는 $\epsilon_t > x_t$가 되어 정보보다 노이즈가 더 많은 형태이다.
그렇기 때문에 결국에는 다음과 같이 모델링이 되어버린다.

$$ y_{t+1} = y_t + \epsilon_t $$


즉, 주어진 정보를 다음의 주가로 반영을 하지 못하고 현재의 주가에 대한 그대로의 반영이 되어 예측을 할 수가 없어진다.

### problem 1. Time-series denoising

noise라기보다는 비선형이라고 하는 것이 더 맞다고 생각한다. 즉, 주가의 그래프를 완벽하게 식으로서 표현할 수가 없다. 그리고 우리가 표현할 수 없는 부분을 우리가 예측하고자 사용하는 수식에서 noise라고 표현할 뿐이다. 그리고 이러한 noise를 해결하는 것이 비선형이 심할수록 더 어려우며, 당연히 그만큼 예측하기도 힘들다. 이에 대해서 다음과 같은 기본적인 해결법이 존재한다.

* Moving Aervage
> 하지만, 단순히 오른쪽으로 lagging 될뿐...

* Bilateral Fiter
> 하지만, lagging은 없어졌으나 결국에는 충분치 않음...

* Auto-Encoder: feature를 추출하기 전에 AE활용하여 CNN Stacked AutoEncoder를 활용하여 denoising module로서 활용

$$ input data -> AE -> feature extraction -> output data $$

이는 feature extraction을 통해 주어진 task의 performance를 내기 위한 output data를 만드는 과정에서 중요한 정보만을 더욱 더 함축하게 됨으로서 noise를 제거할 수 있다는 방법이다.  


### problem 2. 시계열 feature 종류 대비 짧은 시계열 길이

주가의 데이터는 결국에는 과거 주식이 시작할 때부터 현재의 시점까지만 존재한다. 어떻게 보면 긴 시간이라고 할 수도 있지만, 예측을 위한 학습에서의 데이터로서는 많이 부족하다. (예를 들어, 다른 컴퓨터비전이나 자연어처리와 같은 분야와 비교를 한다면 턱없이 부족하다.) 그렇기 때문에 이를 해결하고자 하는 관점은 다음과 같이 두가지가 존재한다. (해결법이 아닌 해결을 위한 관점이다.)

* 기존의 feature를 잘 활용하자 <br/>

예를 들어, 기존의 feature에 중요도가 높은 것에 대하여 attention과 같은 기법을 활용하여 많은 정보에 차이를 줄 수 있다.

* 새로운 데이터를 만들어 보자 <br/>

기존의 데이터를 활용하여 기존의 데이터를 예측하는 데에 도움이 되는 데이터를 생성하는 것이다. 


### problem 3. problem 1과 2로 인한 overfitting

앞서 제시한 problem 1, 2에 의해서 overfitting이 발생한다. 이는 생각보다 치명적이다. overfitting은 problem 1, 2 때문에 발생하는 것이지만, 이를 해결하는 것이 problem 1, 2가 다시 제약조건으로 되어버리기 때문에 더욱 어렵다. 다음은 이를 해결하기 위한 기본적인 방법을 제시한다.

* 딥러닝의 경우 모델의 initialization을 어떻게 하느냐에 따라 도달하는 점이 다름 <br/>

여러 경우의 initailization을 활용하여 더 좋은 initialization의 경우를 선택한다. 

> 개인적인 생각으로는 제일 좋은 것을 추릴수는 있지만, 성능의 향상 또는 overfitting을 해결하는 근본적인 해결책은 아직까지는 아니라고 본다.

또는, AE를 unsupervised learning의 pre-trained net의 weight를 initail weight로 활용

> 오히려 이 방법이 여러 initialization의 경우에 대한 ensemble method보다는 더 효과적이라고 생각한다.


* Bayesian Inference를 활용하여 관찰되지 못한 부분에 대해서 uncertainty를 추정

> Monte Carlo Dropout, Gal, Y., 2016

> Monte Carlo Batch Normalization, Teye, M., 2018

> Deep learning Regression + Gaussian Process Regression 
>> 마지막 단의 노드들을 이용하여 GPR을 활용하여 uncertainty를 추정 


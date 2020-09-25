# quant

> James Simons는 한때 세계 최고의 연봉(약 1조 8척억원!!!)을 받는 헤지 펀드 매니저이다. 원래는 수학교수로서 수학과 물리학에 매우 저명하며, 수학의 공식을 활용하여 주식시장에 투자를 하였다.

#### quant = quantitative + analyst

수학/통계지식을 이용해서 투자법칙을 찾아내 투자모델을 만들거나 금융시장의 변화를 예측하는 사람



#### 주식
* 시가, 고가, 저가, 종가
* 거래량(외국인, 기관, 개인, 프로그램 순매수/순매도)
* 공매도 비율
* 보조지표, 차트패턴

#### 재무제표
* 주가수익비율(Price Earning Ratio, PER) = 주가/주당순이익
* 주가순자산비율(Price Book-value Ratio, PBR) = 주가/주당순자산
* 자기자본이익률(Return On Equity, ROE) = PBR/PER
* 주가현금흐름비율(Price-to-Cash Flow Ratio, PCR) = 주가/주당현금흐름

#### 환율

#### 기타
* 기준 금리 시계열
* 국채 시계열
* 금 시세 시계열
* 유가 시계열
* GDP 시계열
* 경기종합지수 시계열
* 검색 트렌드

### References
* https://www.investing.com/
* https://finance.naver.com/
* http://comp.wisereport.co.kr

------------------------------------------------------------------------------------------------------------
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



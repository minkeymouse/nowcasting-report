3. 혼합 주기 모형

□ TimeMixer
◦ 다양한 주기(sampling scale)에서 분해한 추세와 계절성을 혼합하여 과거 정보를 추출하고, 예측을 위해 앙상블하는 방법으로 혼합주기 예측
◦ 입력 시퀀스를 다운샘플링하여 여러 주기를 표현: 본 실험에서 주간 월간 혼합주기를 반영하기 위해 downsampling_window=4 로 설정
◦ PDM(past decomposable mixing), FMM(future multipredictor mixing) 블록으로 모델을 구분하여 PDM에서 추세와 계절성을 혼합하고 FMM에서 예측치를 앙상블함: X IN  R ^{B TIMES  T TIMES  N} rarrow  [X _{0} ,X _{1} , CDOTS  ,X _{M} ], X _{m} IN  R ^{B TIMES  [T/w ^{m} ] TIMES  N}
◦ B는 배치 크기, T는 시점 수, N은 변수 개수, M은 주기의 개수(다운 샘플링), w는 주기의 단위(다운샘플링 단위), L은 PDM레이어의 수를 의미
◦ 다운샘플링은 평균, 최대값, 컨볼루션 연산을 활용할 수 있음

□ 단기 예측 실험 결과

□ 단기 예측 결과 해석

□ 장기 예측 실험 결과

□ 장기 예측 결과 해석
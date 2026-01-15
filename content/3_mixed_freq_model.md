3. 혼합 주기 모형

□ TimeMixer
◦ 다양한 주기(sampling scale)에서 분해한 추세와 계절성을 혼합하여 과거 정보를 추출하고, 예측을 위해 앙상블하는 방법으로 혼합주기 예측
◦ 주간-월간 혼합주기 데이터를 처리하기 위해 2개 스케일(scale) 사용: 주간(weekly) → 월간(monthly)
◦ 입력 시퀀스를 다운샘플링하여 여러 주기를 표현: 본 실험에서 주간 월간 혼합주기를 반영하기 위해 downsampling_window=4, down_sampling_layers=1 로 설정
◦ PDM(past decomposable mixing), FMM(future multipredictor mixing) 블록으로 모델을 구분하여 PDM에서 추세와 계절성을 혼합하고 FMM에서 예측치를 앙상블함: X IN  R ^{B TIMES  T TIMES  N} rarrow  [X _{0} ,X _{1} , CDOTS  ,X _{M} ], X _{m} IN  R ^{B TIMES  [T/w ^{m} ] TIMES  N}
◦ B는 배치 크기, T는 시점 수, N은 변수 개수, M은 주기의 개수(다운 샘플링), w는 주기의 단위(다운샘플링 단위), L은 PDM레이어의 수를 의미
◦ 다운샘플링은 평균, 최대값, 컨볼루션 연산을 활용할 수 있음


□ 단기 예측 실험 결과
◦ 앞선 실험과 동일한 기간, 동일한 평가지표 및 업데이트 방식 사용
◦ 혼합주기 처리: down_sampling_layers=1, down_sampling_window=4로 주간→월간 2스케일 구조 사용

| 모델명 | 생산 지표 단기 예측 | | 투자 지표 단기 예측 | |
|--------|-------------------|--|-------------------|--|
| | sMSE | sMAE | sMSE | sMAE |
| TimeMixer | 8.61 | 2.20 | 1.53 | 0.97 |


□ 단기 예측 결과 해석
◦ 앞 실험과 동일한 기간, 동일한 평가지표 및 업데이트 방식 사용
◦ 값 자체가 크고 분산이 작은 생산지표의 경우 TimeMixer가 학습할 수 있는 정보가 적어 예측 결과가 좋지 않음
◦ 혼합 주기 mixing 과정에서 과소한 분산으로 인해 노이즈가 자리잡았을 가능성
◦ 혼합 주기 구조는 월간 시계열 비중이 높은 데이터에서 장점이 기대되나, 생산 지표는 수준 대비 변동이 작아 개선 폭이 제한됨
◦ 투자 지표에서는 sMSE/sMAE가 상대적으로 양호해 혼합 주기 구조가 투자 데이터에 더 잘 맞는 양상을 보임
◦ down_sampling_window=4, down_sampling_layers=1 설정과 학습률(0.0005) 조정이 성능에 가장 큰 영향을 준 것으로 확인됨
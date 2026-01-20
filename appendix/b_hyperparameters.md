□ 주요 하이퍼 패러미터 특징은 다음과 같음

◦ 입력 길이(lookback / context_length / n_lags)는 대체로 96주를 사용하였고(TimeMixer는 192주), 주간 데이터에서 약 2년 내의 동학을 포착하면서 결측이 많은 월간 변수(주간 재확장 NaN 포함)에서 학습 윈도우 수가 과도하게 줄어드는 것을 방지하기 위함
◦ 예측 길이(prediction_length / n_forecasts / horizon)는 단기 실험에서 1주 ahead, 장기 실험에서 최대 40주를 기준으로 4~40주 horizon을 동일 모델에서 추출하도록 설정하였고, 실험 설계(단기: 재귀적 1-step, 장기: max horizon 1회 학습 후 다중 horizon 비교)와 일치시키기 위함
◦ 배치 크기(batch_size)는 NeuralForecast 계열에서 주로 128을 사용하고(MAMBA는 32), 다변량 학습의 안정성과 학습 속도를 확보하되 MAMBA는 SSM 블록 및 입력 프로젝션 등으로 메모리 여유를 고려해 보수적으로 설정
◦ 학습 반복(max_steps / max_epochs)은 대략 50 epochs 수준(예: max_steps=550/1100)으로 맞추고 DDFM은 max_epoch=50을 사용하며, 데이터 길이 대비 과적합을 억제하면서도 수렴 여부를 확인 가능한 반복 횟수를 확보하기 위함
◦ 학습률(learning_rate)은 기본 1e-3을 사용하되(TimeMixer 및 일부 DDFM은 5e-4), 혼합주기/믹싱 구조에서 발산이나 진동이 발생할 수 있어 더 작은 학습률로 학습 안정성을 확보하기 위함
◦ 모델 폭(d_model / hidden_size)은 TFT 128, PatchTST 384, iTransformer 512, MAMBA 128~256 수준을 사용하며, 40~42개 다변량에서 표현력 부족과 과적합/불안정 사이의 균형을 맞추기 위함
◦ 모델 깊이(num_layers / e_layers / n_layers)는 PatchTST 6, iTransformer e_layers 3(d_layers 1), TimeMixer 3, MAMBA 4~6 수준을 사용하며, 결측이 많고 유효 신호가 약한 구간에서 과도한 깊이로 인한 불안정을 피하기 위함
◦ 어텐션 헤드 수(n_heads / num_heads)는 PatchTST 16, TFT/iTransformer 8을 사용하며, 변수 간 상관 구조를 여러 서브스페이스로 분해해 학습하되 계산비용과 노이즈 민감도를 고려해 8~16 범위에서 타협
◦ 정규화/규제(dropout, batch norm, RevIN)는 dropout 0.1을 공통적으로 사용하고 DDFM은 인코더에 batch norm을 적용하며 MAMBA는 RevIN을 적용(투자 설정)하고, 분포 변화/스케일 차이 및 과적합을 완화하기 위함
◦ 결측 처리(보간/제한)는 DDFM에서 interpolation_limit=10, direction=both를 사용하고(투자: linear, 생산: spline), NaN이 업데이트/인코더 입력에 직접 들어가면 학습/예측이 불안정해질 수 있어 보간을 사용하되 과도한 외삽을 제한하기 위함
# DL 금융 행동 패턴 분석을 통한 신용 등급 분류

고객의 금융 정보를 바탕으로 신용 등급을 분류하는 DNN 기반 3-class 분류 모델



| 항목 | 내용 |
|------|------|
| 목표 | 고객 금융 데이터로 신용 등급 예측 |
| 모델 | Deep Neural Network (DNN) |
| 프레임워크 | PyTorch |
| 최종 성능 | Validation Accuracy **79.65%** |
| 데이터 출처 | [Kaggle - Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) |
| 클리닝 참고 | [Kaggle Notebook - Data Cleaning](https://www.kaggle.com/code/clkmuhammed/credit-score-classification-part-1-data-cleaning) |




## 1. 전처리 (Preprocessing)

### 처리 단계

| 단계 | 처리 내용 |
|------|-----------|
| 1 | 불필요 식별자 제거 |
| 2 | 음수값 방어 처리 |
| 3 | 이상치 클리핑 |
| 4 | `Credit_Mix` — Ordinal Encoding |
| 5 | `Type_of_Loan` — 대출 종류 개수로 수치 변환|
| 6 | `Occupation`, `Payment_Behaviour`, `Payment_of_Min_Amount` — One-Hot Encoding |
| 7 | `Credit_Score` — Target Encoding |
| 8 | RobustScaler 스케일링 |

### 설계 근거

**식별자 제거**
 불필요 식별자 제거를 제거하여 학습에 필요한 변수만 사용
Month는 시계열 특성을 활용하지 않는 구조이므로 제거

**음수 방어 처리**
EDA에서 음수 불가능 변수 15개에 대해 음수값 존재 여부를 사전 확인
확인 결과 음수값은 없었으나, 향후 데이터 변동에 대비해 방어적 처리 구조를 코드에 유지

**이상치 클리핑**
Box Plot(EDA 3)에서 Annual_Income, Monthly_Inhand_Salary 등 다수 변수에서 상위 극단값이 관찰
극단값은 모델 학습 안정성을 저하시킬 수 있어 상위 1% 기준으로 클리핑을 적용

**Credit_Mix Ordinal Encoding**
Good > Standard > Bad의 순서 관계가 명확한 순서형 변수
순서 정보를 보존하기 위해 One-Hot 대신 Ordinal Encoding(2 / 1 / 0)을 적용

**Type_of_Loan 수치 변환**
원본값이 콤마로 구분된 다중값 문자열 구조
모델 입력이 가능하도록 대출 종류의 수(len)로 변환

**RobustScaler**
이상치 클리핑 후에도 잔존 극단값의 영향이 남을 수 있음
평균·표준편차 기반의 StandardScaler 대신 IQR 기반의 RobustScaler를 선택해 이상치 영향을 최소화

---

## 2. EDA (탐색적 데이터 분석)

### EDA 1 — 타겟 클래스 분포
<div align="center">

<img width="590" height="390" alt="target_distribution" src="https://github.com/user-attachments/assets/3790d638-257d-4672-b42a-71fb545e1d8f" />

</div>
`stratify=y` 적용 근거로 클래스 분포 확인

- Standard 클래스 비율이 가장 높고, Good 클래스 비율이 가장 낮은 클래스 비율 차이 확인
- 비율 차이가 있으므로 단순 랜덤 분할 시 학습/검증 세트의 클래스 구성이 달라질 수 있음
- `train_test_split`에 `stratify=y`를 적용해 분할 후에도 원본 비율을 유지

---

### EDA 2 — 음수값 발생 건수 확인

도메인상 음수가 불가능한 변수 15개를 선별해 실제 음수값 존재 여부 확인

- 확인 결과 해당 변수에 음수값 없음
- 전처리 코드는 향후 데이터 변동에 대비해 방어적 처리 구조로 유지

---

### EDA 3 — 수치형 변수 이상치 분포 (Box Plot)

<img width="1589" height="691" alt="boxplot_outliers" src="https://github.com/user-attachments/assets/2a21700f-7cb4-4c4a-be6c-07248a0497cb" />

상위 1% 클리핑 적용 근거로 Box Plot을 통해 극단값 존재 여부 확인

- 수치형 변수 전반에서 상위 극단값(outlier)이 다수 관찰됨
- 특히 Annual_Income, Monthly_Inhand_Salary에서 두드러지게 나타남
- 극단값은 모델 gradient를 불안정하게 만들 수 있어 `quantile(0.99)` 기준 클리핑 적용

---

### EDA 4 — 수치형 변수 상관관계 히트맵

<img width="1116" height="890" alt="correlation_heatmap" src="https://github.com/user-attachments/assets/be42d2e4-e623-4beb-b7ba-0312a03862bc" />

변수 간 관계 구조와 다중공선성을 확인.

- **Annual_Income ↔ Monthly_Inhand_Salary**: 매우 높은 양의 상관관계 확인. 월급과 연소득이 직접 연결된 변수로, 사실상 같은 정보를 담고 있음
- **Interest_Rate ↔ Num_of_Loan ↔ Delay_from_due_date**: 양의 상관관계 확인. 대출 수와 연체 기록이 증가할수록 금리도 함께 높아지는 경향

---

## 3. 모델링 (Modeling)

### 모델 구조

```
Input (수치형 + 범주형 혼합 피처)
    |
    Linear(input_dim → 512) → BatchNorm → ReLU → Dropout(0.3)
    |
    Linear(512 → 256)       → BatchNorm → ReLU → Dropout(0.2)
    |
    Linear(256 → 128)       → BatchNorm → ReLU
    |
    Linear(128 → 3)                              ← 3-class Output
```

### 모델 선택 근거

수치형 변수와 One-Hot 인코딩된 범주형 변수가 혼합된 고차원 구조로, 변수 간 비선형 관계가 존재할 가능성 높음
비선형 패턴을 레이어별로 학습할 수 있는 DNN을 선택

| 설계 요소 | 선택 | 근거 |
|-----------|------|------|
| 초기화 | He Initialization | ReLU 사용 시 기울기 소실 방지에 최적화된 초기화 방식 |
| 정규화 | BatchNorm | 레이어 간 분포 안정화 및 수렴 속도 향상 |
| 과적합 방지 | Dropout (0.3 / 0.2) | 학습 데이터 과적합 방지 |
| 손실 함수 | CrossEntropyLoss | 다중 클래스 분류에 적합한 손실 함수 |

### 학습 설정

| 항목 | 값 | 근거 |
|------|----|------|
| Optimizer | Adam (`lr=0.001`, `weight_decay=1e-5`) | weight decay로 L2 정규화 효과 부여 |
| Scheduler | CosineAnnealingLR (`T_max=10`) | 학습률 점진적 감소로 안정적 수렴 유도 |
| Early Stopping | `patience=10`, Val Accuracy 기준 | 과적합 시점에서 자동 종료 및 최적 모델 저장 |
| Batch Size | 256 | 학습 안정성 및 메모리 고려 |
| Max Epoch | 100 | 모델 학습 시간 고려 |
| Train / Val 분할 | 8 : 2 (`stratify` 적용) | 클래스 비율 유지 |

---

## 4. 성능 결과

| 지표 | 값 |
|------|----|
| Best Validation Accuracy | **79.65%** |

매 epoch마다 Train Loss / Val Loss / Val Accuracy를 계산하고, 5 epoch 주기로 출력해 학습 추이를 모니터링
Early Stopping 발생 시 종료 epoch와 Best Val Accuracy 출력

<img width="810" height="704" alt="화면 캡처 2026-05-12 134225" src="https://github.com/user-attachments/assets/bcaf6b2f-340c-4b81-8fc7-0fcbf2250397" />


---

## 5. 개선사항

**1. Feature Selection 정량화 미적용**
현재 변수 제거는 도메인 기반 수동 선택 방식
Feature Importance, SHAP, 상관계수 기반의 정량적 선택 기법 도입을 고려

**2. 클래스 불균형 처리 심화**
stratify로 클래스 비율만 유지한 상태
클래스 가중치(class_weight) 또는 오버샘플링(SMOTE)을 적용하면 소수 클래스 학습을 강화

**3. 앙상블 미적용**
DNN 단일 모델만 사용
XGBoost, LightGBM 등 트리 기반 모델과의 앙상블로 성능 향상을 기대

**4. 이상치 하한 처리 미흡**
현재 상위 1%만 클리핑을 적용
하위 이상치도 `quantile(0.01)` 기준의 대칭 처리를 검토

**5. 하이퍼파라미터 탐색 미적용**
학습률, Dropout 비율, 레이어 구조 등을 Optuna 기반 자동 탐색으로 최적화 가능

---

## 6. Reference

- Dataset: [Kaggle — Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)
- Data Cleaning: [Kaggle Notebook — Credit Score Classification Part 1: Data Cleaning](https://www.kaggle.com/code/clkmuhammed/credit-score-classification-part-1-data-cleaning)
- PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
- scikit-learn Documentation: [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)

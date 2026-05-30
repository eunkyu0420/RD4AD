# PROMPTS.md - AI 코딩 툴 프롬프트 로그

**사용 도구:** Claude Code (Anthropic, claude-sonnet-4-6 모델)

**과제:** Anomaly Detection via Reverse Distillation from One-Class Embedding (RD4AD) 논문 오픈소스 코드 구현 및 실험 재현

**작성일:** 2026-05-26

---

## 1. 과제 파악 및 계획 수립

**프롬프트:**
> CVPR 2022 논문 "Anomaly Detection via Reverse Distillation from One-Class Embedding"의 오픈소스 코드를 구현하고 실험을 재현해보려고 해. 원본 코드는 https://github.com/hq-deng/RD4AD 에 있어. MVTec-AD 데이터셋으로 논문의 실험 파트를 재현하고 싶은데, 어떤 순서로 진행하면 좋을지 계획을 잡아줘.

**Claude Code 응답 요약:**
- 4단계 계획 수립:
  1. 원본 코드 클론 및 구조 파악
  2. 환경 세팅 (conda 가상환경, 패키지 설치)
  3. MVTec-AD 데이터셋 다운로드 및 배치
  4. Smoke Test (carpet 1 epoch) → 결과 확인
- 제출물: 보고서 + PROMPTS.md

---

## 2. 원본 코드 클론 및 구조 파악

**프롬프트:**
> 원본 코드를 클론해서 어떤 파일들이 있는지 파악하고, 각 파일의 역할을 정리해줘. 특히 논문의 Reverse Distillation 구조가 코드에서 어떻게 구현되어 있는지 설명해줘.

**Claude Code 수행 작업:**
1. GitHub 레포 클론 (`https://github.com/hq-deng/RD4AD`)
2. 전체 파일 목록 출력 및 역할 정리

**응답 요약:**

| 파일 | 역할 |
|------|------|
| `main.py` | 학습 루프, 손실 함수(loss_fucntion), 진입점 |
| `resnet.py` | Teacher Encoder (Wide ResNet50) + BN_layer (OCBE 모듈) |
| `de_resnet.py` | Student Decoder (역방향 ResNet 구조) |
| `dataset.py` | MVTecDataset, 데이터 전처리 transform |
| `test.py` | 평가 함수 (AUROC, AUPRO), anomaly map 계산 |

- 논문의 Reverse Distillation 구조:
  - Teacher: `wide_resnet50_2` (frozen, pretrained on ImageNet)
  - OCBE: `BN_layer` (MFF + OCE 블록, trainable)
  - Student: `de_wide_resnet50_2` (decoder 구조, trainable)
  - 손실 함수: Cosine Similarity Loss (`loss_fucntion` in main.py)

---

## 3. 환경 세팅

**프롬프트:**
> 논문 README에 있는 환경 요구사항으로 conda 가상환경을 세팅해줘. pytorch==1.9.1, torchvision==0.10.1, numpy==1.20.3, scipy==1.7.1, sklearn==1.0, PIL==8.3.2가 필요해. RTX 5070 Ti 환경이야.

**Claude Code 수행 작업:**
```bash
conda create -n rd4ad python=3.8
conda activate rd4ad
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy scikit-learn Pillow
pip install opencv-python matplotlib pandas scikit-image
```

**응답 요약:**
- RTX 5070 Ti는 CUDA 12.x 필요 → 논문 원본(PyTorch 1.9.1, CUDA 11.1) 그대로 설치 불가
- CUDA 12.1 지원 PyTorch 최신 버전으로 설치

---

## 4. MVTec-AD 데이터셋 준비

**프롬프트:**
> MVTec-AD 데이터셋을 다운로드해서 코드가 읽을 수 있는 구조로 배치하는 방법을 알려줘. 폴더 구조가 어떻게 되어야 해?

**Claude Code 응답 요약:**
- 공식 사이트 (https://www.mvtec.com/company/research/datasets/mvtec-ad) 에서 다운로드
- 압축 해제 후 아래 구조로 배치:

```
RD4AD/
├── mvtec/
│   ├── carpet/
│   ├── bottle/
│   └── ... (총 15개 클래스)
├── main.py
├── dataset.py
├── resnet.py
├── de_resnet.py
└── test.py
```

---

## 5. main.py 수정 — argparse 인자 추가

**프롬프트:**
> 전체 15개 클래스를 200 epoch씩 돌리면 시간이 너무 오래 걸릴 것 같아. 먼저 carpet 클래스만 1 epoch로 동작 확인을 해보고 싶은데, main.py에 --class_name이랑 --epochs 인자를 추가해줘.

**Claude Code 수행 작업:**

원본 대비 변경된 내용:

1. `train(_class_)` → `train(_class_, epochs=200)` 시그니처 변경, 내부 `epochs = 200` 제거
2. `--class_name`, `--epochs` argparse 인자 추가
3. `os.makedirs('./checkpoints', exist_ok=True)` 추가 (폴더 자동 생성)
4. 마지막 epoch 평가 조건 추가: `if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:`

```python
# 원본
def train(_class_):
    epochs = 200
    ...
    if (epoch + 1) % 10 == 0:
        ...
    for i in item_list:
        train(i)

# 수정 후
def train(_class_, epochs=200):
    ...
    os.makedirs('./checkpoints', exist_ok=True)
    if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
        ...
    parser = argparse.ArgumentParser(description='RD4AD Anomaly Detection')
    parser.add_argument('--class_name', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()

    if args.class_name is not None:
        train(args.class_name, epochs=args.epochs)
    else:
        for i in item_list:
            train(i, epochs=args.epochs)
```

---

## 6. README.md 업데이트

**프롬프트:**
> README를 한국어로 정리하고, 추가한 --class_name, --epochs 인자 사용법이랑 폴더 구조도 넣어줘.

**Claude Code 수행 작업:**
- 원본 영문 README를 한국어로 재작성
- 폴더 구조 다이어그램 추가
- `--class_name`, `--epochs` 인자 설명 표 추가
- Smoke Test 실행 명령어 및 결과 섹션 추가
- `checkpoints/` 폴더 자동 생성 안내 추가

---

## 7. Smoke Test 실행

**프롬프트:**
> 수정한 main.py로 carpet 클래스 1 epoch Smoke Test를 실행해줘.

**실행 명령:**
```bash
python main.py --class_name carpet --epochs 1
```

**실행 환경:** PyTorch + CUDA (RTX 5070 Ti), rd4ad Conda 가상환경

**결과:**

| 지표 | 결과 |
|------|------|
| Pixel Auroc | 0.993 |
| Sample Auroc | 1.000 |
| Pixel Aupro | 0.978 |

→ 모델 구조 및 데이터 파이프라인 정상 동작 확인.

---

## 최종 산출물 요약

| 항목 | 내용 |
|------|------|
| 수정된 파일 | `main.py` (argparse 추가, os.makedirs, 마지막 epoch 평가 조건) |
| 새로 작성된 파일 | `README.md` (한국어 재작성, 사용법/결과 추가) |
| Smoke Test 결과 | carpet 1 epoch: Pixel AUROC 0.993, Sample AUROC 1.000, AUPRO 0.978 |

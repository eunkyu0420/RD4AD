# CVPR2022 - Anomaly Detection via Reverse Distillation from One-Class Embedding
## Implementation (Official Code ⭐️ ⭐️ ⭐️)

> **원본 저장소**: https://github.com/hq-deng/RD4AD

---

## 1. Environment

- pytorch == 1.9.1
- torchvision == 0.10.1
- numpy == 1.20.3
- scipy == 1.7.1
- sklearn == 1.0
- PIL == 8.3.2

---

## 2. Dataset

MVTec-AD 데이터셋을 [MVTec 공식 사이트](https://www.mvtec.com/company/research/datasets/mvtec-ad/)에서 다운로드 후, 코드 폴더 내 `mvtec/` 폴더에 압축 해제합니다.

```
RD4AD/
├── mvtec/
│   ├── carpet/
│   ├── bottle/
│   └── ...
├── main.py
├── dataset.py
├── resnet.py
├── de_resnet.py
└── test.py
```

---

## 3. Train and Test

### 전체 실행 (원본 방식 - 15개 클래스 × 200 epoch)
```bash
python main.py
```

### 단일 클래스 / 지정 epoch 실행
```bash
# 예시: carpet 클래스만 1 epoch (Smoke Test)
python main.py --class_name carpet --epochs 1

# 예시: carpet 클래스만 200 epoch
python main.py --class_name carpet --epochs 200
```

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--class_name` | 학습할 클래스 지정 (미지정 시 전체 15개 실행) | None |
| `--epochs` | 학습 epoch 수 | 200 |

> ✅ `checkpoints/` 폴더는 실행 시 자동 생성됩니다.

---

## 4. Smoke Test 결과

컴퓨팅 자원 및 시간 제약으로 인해 `carpet` 클래스 단일 항목, 1 epoch으로 동작 검증을 수행했습니다.

```bash
python main.py --class_name carpet --epochs 1
```

**실행 환경**: PyTorch + CUDA (RTX 5070 Ti), rd4ad Conda 가상환경

| 지표 | 결과 |
|------|------|
| Pixel Auroc | 0.993 |
| Sample Auroc | 1.000 |
| Pixel Aupro | 0.978 |

---

## 5. Reference

```
@InProceedings{Deng_2022_CVPR,
    author    = {Deng, Hanqiu and Li, Xingyu},
    title     = {Anomaly Detection via Reverse Distillation From One-Class Embedding},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {9737-9746}
}
```

# Mac Accelerator

[English](README.md) · [한국어](README.ko.md)

Apple Silicon / Core ML 추론을 연구하는 **독립형 macOS 앱과 서버**입니다.
sunnypilot M2 실험에서 분리했으며, 빌드·실행에 sunnypilot 체크아웃,
tinygrad, 기기 펌웨어 또는 차량 제어 코드는 필요하지 않습니다.

macOS 15 이상 · arm64 · M2 시험 · 연구용 프리뷰 0.3.0

앱은 Core ML CPU + Neural Engine 서버와 인증된 합성 클라이언트를 실행합니다.
실차용 가속기, macOS 외장 GPU 드라이버 또는 임의 모델 실행기가 아닙니다.
모델 계약은 기존 실험의 Big Model 주행 정책 프로토콜입니다. 앱에는 실시간
3X 카메라 연결, 기기 파라미터 쓰기, 차량 제어 출력 게시 기능이 없습니다.

## 현재 우선순위: 지연시간보다 전원 안정성

2026-09-29: 맥 USB 직접 연결 구성에서 3X 재부팅이 반복됐고, sunnypilot
master 복원 후에도 UVLO/SMPL 전원 리셋 표시가 기록됐습니다. 운용 관찰에서는
맥 분리 후와 차량 분리 환경에서 재부팅이 보고되지 않았습니다.
보고된 실패 조건은 차량 + 맥 USB 연결 조합입니다. 비교 시간과
부하는 일치시킨 실험이 아닙니다. **정확한 원인과 해결책은 미확정**입니다.
차량에서 직접 USB 연결을 재시험하거나 과거 성능 수치를 실차 검증으로 해석하지 마세요.

- 연구 요약: [한국어](docs/RESEARCH_SUMMARY.ko.md) · [English](docs/RESEARCH_SUMMARY.md)

프로젝트 소개와 최신 연구 요약은 한·영으로 제공합니다. 과거 기술 보고서는
원래 언어로 보존합니다.

## 설치 및 모델 준비

[앱 프리뷰 0.3.0](https://github.com/Akarow77/mac-accelerator/releases/tag/v0.3.0)을
받거나 소스에서 빌드할 수 있습니다. 앱은 ad-hoc 서명이며 Apple 공증을 받지
않았습니다. ZIP만으로 실행 환경과 모델까지 설치되지는 않습니다. Gatekeeper를
전역 해제하지 말고, 필요하면 소스를 검토하고 직접 빌드하세요.

```bash
git clone https://github.com/Akarow77/mac-accelerator.git
cd mac-accelerator
./setup_coreml_env.sh
```

`uv`로 별도 Python 3.12 `.coreml-venv`와 고정된 런타임 의존성을 설치합니다.
모델, 실행 환경, 개인 로그는 Git에서 제외합니다.

신뢰할 수 있는 Big Model ONNX와 **정확히 대응하는 메타데이터**를 직접 준비하세요.
모델을 재배포하거나 자동 다운로드하지 않습니다. 예시 경로를 실제 경로로 바꿉니다.

```bash
.coreml-venv/bin/python import_model.py \
  --onnx /path/to/big_driving_supercombo.onnx \
  --metadata /path/to/big_driving_supercombo_metadata.pkl \
  --coreml /path/to/big_driving.mlpackage
```

기존 파일을 덮어쓰거나 심볼릭 링크를 만들지 않고 독립 복사합니다. ONNX SHA-256은
출처 식별용이지 변환 정확도의 증명이 아닙니다. 메타데이터 pickle은 코드 실행이
가능하므로 출처 불명의 파일은 사용하지 마세요. Git LFS 포인터도 실제 모델이 아닙니다.
모델에는 해당 모델의 이용 조건이 적용됩니다.

변환 패키지가 없다면 `--coreml`을 생략하고 다음을 실행합니다.

```bash
./setup_coreml_env.sh --conversion
./compile_coreml.sh
```

변환에는 시간과 메모리가 필요합니다. 임의 ONNX의 메타데이터를 생성하는 기능은
없으며, 변환 성공만으로 추론 정확도가 검증되지 않습니다.

## 앱 빌드와 맥 단독 시험

Apple Command Line Tools가 필요합니다.

```bash
./build_macos_app.sh
open "dist/Mac Accelerator.app"
```

앱에서 이 프로젝트 폴더를 선택합니다. **Mac-only test가 기본값**입니다.
Ready / Client connected는 서버 상태이지 주행 준비 상태가 아닙니다.
앱 식별자는 `dev.akarow.mac-accelerator`이며 인증 키는
`~/Library/Application Support/Mac Accelerator/auth.key`에 따로 보관합니다.
이전 앱의 설정이나 키를 재사용·삭제하지 않습니다. 포트 8066 충돌을 피하려면
이전 서버를 먼저 종료하고, 클라이언트도 새 키를 사용해야 합니다.

별도 서버 없이 합성 시험을 실행하려면:

```bash
.coreml-venv/bin/python smoke_local.py --frames 200 \
  --shadow-log artifacts/shadow-test-001.jsonl
```

매번 새 로그 경로를 사용하세요. 실제 녹화 입력 재생, 연결 끊김 주입, 수동
클라이언트 실행 방법은 [영문 실행 안내](README.md#observation-only-shadow-replay)에
있습니다. 재생은 맥 측 관찰 전용이며 실시간 3X 카메라 수신이 아닙니다.
워프 입력은 uint8 `[frames,2,6,128,256]`, 정책 입력은 대응하는 float32
`[frames,12]`여야 합니다. 입력을 임의로 합성해 녹화 데이터라고 취급하지 않습니다.

추론만 측정할 때는 앱 서버를 먼저 종료하고 실행합니다.

```bash
DURATION=30 ./run_coreml_qualification.sh
```

USB·카메라 수신은 제외됩니다. 팬 없는 MacBook Air는 장시간 시험에서 뜨거워질
수 있습니다. 시간 문맥은 연속 132프레임(간격 4, 33단계) 준비 후 평가합니다.
50ms 초과를 집계하며, 150ms stale-stop은 통과 기준이 아닙니다.

## 선택적 기능과 제한

- 사전 적재 NV12 전처리와 tinygrad/Metal 비교는 [전처리 연구](docs/CAMERA_PIPELINE.md)를
  참고하세요. 최신 M2 선택은 사전 계산된 Metal 참조 좌표를 사용하는 native CPU
  gather + Core ML CPU/NE였습니다. GPU/ANE 분할 실험은 앱·서버에 통합하지 않았습니다.
- 선택적 USB 모드는 ADB/NCM 설정을 사용하지만 앱이 펌웨어나 기기 파라미터를
  변경하지는 않습니다. 전원 문제가 열린 동안 차량 연결 사용은 권장하지 않습니다.
- `tools/`의 카메라 시작·CPU 유지·파일 staging 등은 별도 명시적 벤치 도구입니다.
  앱에서 자동 실행되지 않습니다. offroad 표시만으로 차량 분리를 확인할 수 없습니다.
- [그림자 재생 경계](docs/SHADOW_DESIGN.md), [프로토콜](docs/PROTOCOL.md),
  [0.3.0 검증 기록](docs/RELEASE_0.3.0.md), [과거 진행 기록](docs/PROJECT_STATUS.md)을 참고하세요.

```bash
.coreml-venv/bin/python -m unittest discover -s . -p 'test_*.py'
```

과거 맥 단독 6,000회 시험의 추론 평균 25.69ms / 최대 39.53ms는 역사적 수치입니다.
실제 카메라부터 결과까지의 지속적인 20Hz 경로는 검증되지 않았습니다. M1 및 다른
M칩의 성능·안정성도 M2 결과로 보증하지 않습니다. 전원, 문맥 정확성, 장시간 지연,
로컬 모델 공존을 검증할 때까지 벤치 연구로 유지합니다.

By Codex

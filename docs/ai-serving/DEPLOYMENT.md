# 실행 및 배포 준비 현황

## 로컬

Python 3.12, CatBoost 1.2.10 환경에서 저장소 루트에서 실행합니다. `deployment/ai/requirements.txt`는 현재 로컬에서 확인한 버전이며 깨끗한 환경 및 Linux 이미지 설치는 아직 검증하지 않았습니다.

```powershell
python -m pip install -r deployment/ai/requirements.txt
python -m uvicorn kbo_bandit.serving.api:app --host 127.0.0.1 --port 8000
```

원본 동결 폴더가 필요합니다. `KBO_B1_ARTIFACT_DIR`, `KBO_CATBOOST_SERVICE_ARTIFACT_DIR`로 다른 위치를 지정할 수 있습니다. `/health`는 B1만 확인하므로 v6 예측도 별도로 확인하세요. v5에는 `KBO_CATBOOST_ARTIFACT_DIR`, `KBO_CATBOOST_WEB_MAPPING`이 별도로 필요합니다.

## Docker 컨텍스트

생성기는 지정한 **새 폴더**에만 실행되며 일곱 모델 파일과 명시된 서빙 코드만 복사하고, 고정 SHA-256을 복사 전후 확인합니다. 학습 parquet, B1 ETL snapshot 및 v5 모델은 이미지에 포함되지 않습니다. 모델 소스와 별도로 받은 파일의 신뢰성·배포 권한은 팀에서 확인해야 합니다.

```powershell
python deployment/ai/build_context.py outputs/ai-context
docker build -t nemosdream-ai:local outputs/ai-context
docker run --rm --name nemosdream-ai -p 127.0.0.1:8000:8000 nemosdream-ai:local
Invoke-RestMethod http://localhost:8000/health
```

```bash
python deployment/ai/build_context.py outputs/ai-context
docker build -t nemosdream-ai:local outputs/ai-context
docker run --rm --name nemosdream-ai -p 127.0.0.1:8000:8000 nemosdream-ai:local
curl http://localhost:8000/health
```

컨테이너 시작 시 두 모델의 버전과 해시를 확인한 후 0.0.0.0:8000에 바인딩합니다. 호스트 공개 포트는 로컬 예제에만 사용합니다. 실제 EC2에서는 기존 배포를 변경하기 전에 Spring과 AI 컨테이너를 동일한 사설 Docker 네트워크에 연결하고 Spring 주소를 `http://nemosdream-ai:8000`으로 설정하며 AI 포트의 외부 게시를 피합니다. 네트워크, 리소스, 이미지·아티팩트 보관 및 릴리스 권한은 아직 확정되지 않았습니다.

Docker 엔진을 사용할 수 없어 빌드와 컨테이너 HTTP는 아직 검증하지 않았습니다. 배포 가능하다고 확정할 수 없습니다. 저장소가 공개인지와 데이터·모델 배포 권한을 확인한 후 접근 통제된 릴리스 또는 객체 저장소 등을 선택하고 해시와 함께 전달하세요. Git ignore된 `outputs/`만으로는 팀원이 모델을 받을 수 없습니다.

이 작업 공간에서는 `python deployment/ai/smoke_local.py`로 B1 변환 5건, v6 golden 128건, localhost HTTP·입력 오류·모델 누락 및 손상을 확인했습니다. `python deployment/ai/smoke_context.py outputs/ai-context-verified-2`는 별도 프로세스에서 학습 Parquet 접근을 차단하고 생성된 컨텍스트의 두 HTTP API를 확인했습니다. 이는 Docker 검증을 대신하지 않습니다. smoke 스크립트에는 개발 검증용 `httpx==0.28.1`이 추가로 필요하며 런타임 이미지 의존성에는 포함하지 않았습니다.

추론에 필요한 파일은 생성된 `artifact-manifest.json`에 크기와 SHA-256이 기록됩니다. B1 ETL의 `pitcher_snapshots.csv` (4,192,475 bytes), `b1_train_frozen_li_lookup.csv` (205,913 bytes), v6의 `pitch_count_policy.json` (4,512 bytes)은 이미지가 아닌 Spring 측 데이터·정책 전달 대상으로 별도 배포가 필요합니다. 2023 고정 이력은 모델의 `train_history_2023.json`에 포함되며 선택적으로 CSV 두 개를 DB에 적재할 수 있습니다.

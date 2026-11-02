# 실행 및 배포 준비 현황

## 로컬

Python 3.12, CatBoost 1.2.10 환경에서 저장소 루트에서 실행합니다. `deployment/ai/requirements.txt`의 직접 의존성은 로컬과 Linux Docker 이미지에서 확인했습니다. 전이 의존성과 `python:3.12-slim` 태그는 고정되어 있지 않으므로 향후 빌드에서는 달라질 수 있습니다.

```powershell
python -m pip install -r deployment/ai/requirements.txt
python -m uvicorn kbo_bandit.serving.api:app --host 127.0.0.1 --port 8000
```

원본 동결 폴더가 필요합니다. `KBO_B1_ARTIFACT_DIR`, `KBO_CATBOOST_SERVICE_ARTIFACT_DIR`로 다른 위치를 지정할 수 있습니다. `/health`는 B1만 확인하므로 v6 예측도 별도로 확인하세요. v5에는 `KBO_CATBOOST_ARTIFACT_DIR`, `KBO_CATBOOST_WEB_MAPPING`이 별도로 필요합니다.

## Docker 컨텍스트

생성기는 지정한 **새 폴더**에만 실행되며 일곱 모델 파일과 명시된 서빙 코드만 복사하고, 고정 SHA-256을 복사 전후 확인합니다. 학습 parquet, B1 ETL snapshot 및 v5 모델은 이미지에 포함되지 않습니다. 모델 소스와 별도로 받은 파일의 신뢰성·배포 권한은 팀에서 확인해야 합니다.

다른 폴더·이미지·컨테이너와 충돌하지 않도록 매번 고유한 이름과 로컬 임시 포트를 사용합니다.

```powershell
$runId = [guid]::NewGuid().ToString('N').Substring(0, 12)
$context = "outputs/ai-context-$runId"
$image = "nemosdream-ai:local-$runId"
$container = "nemosdream-ai-$runId"
python deployment/ai/build_context.py $context
docker build -t $image $context
docker run -d --name $container -p '127.0.0.1::8000' $image
$address = docker port $container 8000/tcp
Invoke-RestMethod "http://$address/health"
# API_CONTRACT.md의 두 요청도 확인한 후 이 테스트 컨테이너만 중지합니다.
docker stop $container
```

```bash
run_id="$(date +%s)-$$"
context="outputs/ai-context-$run_id"
image="nemosdream-ai:local-$run_id"
container="nemosdream-ai-$run_id"
python deployment/ai/build_context.py "$context"
docker build -t "$image" "$context"
docker run -d --name "$container" -p 127.0.0.1::8000 "$image"
address=$(docker port "$container" 8000/tcp | head -n 1)
curl "http://$address/health"
# API_CONTRACT.md의 두 요청도 확인한 후 이 테스트 컨테이너만 중지합니다.
docker stop "$container"
```

컨테이너 시작 시 두 모델의 버전과 해시를 확인한 후 0.0.0.0:8000에 바인딩합니다. 호스트 공개 포트는 로컬 예제에만 사용합니다. 실제 EC2에서는 기존 배포를 변경하기 전에 Spring과 AI 컨테이너를 동일한 사설 Docker 네트워크에 연결하고 Spring 주소를 `http://nemosdream-ai:8000`으로 설정하며 AI 포트의 외부 게시를 피합니다. 네트워크, 리소스, 이미지·아티팩트 보관 및 릴리스 권한은 아직 확정되지 않았습니다.

2026-11-02 Docker Desktop Linux 환경에서 새 컨텍스트와 고유 이미지 태그로 캐시 없는 빌드 및 의존성 설치를 확인했습니다. 7개 모델 파일은 원본·복사본 모두 SHA-256이 일치했습니다. 컨테이너에서 B1과 v6 HTTP 응답을 동결 기준과 비교하고 잘못된 요청, 반복 요청 및 재시작 후 두 API를 검증했습니다. 컨테이너 내부 임시 복사본을 대상으로 B1·v6 각각 모델 파일 누락과 해시 손상 시 `startup.py`가 서버 시작 전에 오류로 종료하는 것도 확인했습니다. 이는 실제 EC2 배포 검증은 아닙니다.

모델 전달 위치·권한과 운영 배포는 아직 확정되지 않았습니다. 이 저장소는 공개 저장소이므로 공개 GitHub Release에 모델을 올리기 전에 배포 권한을 확인해야 합니다. 팀 계정으로 접근이 제한된 공유 드라이브 또는 비공개 객체 저장소에 전달하고, 별도 채널로 묶음 SHA-256을 공유하세요. Git ignore된 `outputs/`만으로는 팀원이 모델을 받을 수 없습니다.

이 작업 공간에서는 `python deployment/ai/smoke_local.py`로 B1 변환 5건, v6 golden 128건, localhost HTTP·입력 오류·모델 누락 및 손상을 확인했습니다. `python deployment/ai/smoke_context.py outputs/ai-context-verified-2`는 별도 프로세스에서 학습 Parquet 접근을 차단하고 생성된 컨텍스트의 두 HTTP API를 확인했습니다. 이는 Docker 검증을 대신하지 않습니다. smoke 스크립트에는 개발 검증용 `httpx==0.28.1`이 추가로 필요하며 런타임 이미지 의존성에는 포함하지 않았습니다.

추론에 필요한 파일은 생성된 `artifact-manifest.json`에 크기와 SHA-256이 기록됩니다. B1 ETL의 `pitcher_snapshots.csv` (4,192,475 bytes), `b1_train_frozen_li_lookup.csv` (205,913 bytes), v6의 `pitch_count_policy.json` (4,512 bytes)은 이미지가 아닌 Spring 측 데이터·정책 전달 대상으로 별도 배포가 필요합니다. 2023 고정 이력은 모델의 `train_history_2023.json`에 포함되며 선택적으로 CSV 두 개를 DB에 적재할 수 있습니다.

팀 전달용 로컬 묶음은 모델 7개, 위 Spring 연동 데이터와 v6 이력 CSV, 선택적 smoke 검증 자료를 허용 목록으로 담고 각 파일의 상대 경로·크기·SHA-256을 `outputs/ai-handoff-manifest.json`에 기록합니다. 전달자는 묶음 전체 해시와 배포 권한을 확인하고, 수신자는 파일 해시를 검증한 뒤 충돌할 `outputs/`가 없는 새 체크아웃에 압축을 풉니다. B1 Snapshot과 TRAIN LI는 2025 시즌 기준입니다. 기존 모델 파일을 덮어쓰지 마세요.

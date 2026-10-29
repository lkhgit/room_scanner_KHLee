# Room Scanner

Local web app that combines the sunlight calculator, mold simulator and flood-risk map for one scanned room.

    run.bat            (or: pip install -r requirements.txt; python server.py)  ->  http://localhost:8000

## 실행 방법

### 1. 서버 켜기
PowerShell에서:

    cd app\room_scanner
    py -m pip install -r requirements.txt     # 처음 한 번만
    py server.py

브라우저에서 http://localhost:8000 을 연다. 서버를 끄려면 PowerShell에서 Ctrl+C.

- `python` 대신 `py`를 쓴다. 이 PC에서는 `python`이 pip 없는 MSYS2 Python(`C:\msys64\ucrt64\bin`)으로
  잡혀서 `No module named pip` 오류가 난다. 같은 이유로 `run.bat`도 그대로는 실패한다.
- 반드시 `app/room_scanner` 폴더 안에서 실행한다 (상위 폴더에서는 `server.py`를 찾지 못한다).
- 코드를 고친 뒤에는 서버를 껐다가 다시 켠다.

### 2. 방 스캔하기 (Home -> Start Room Scan)
두 모듈을 순서대로 하나씩 USB로 연결한다. 펌웨어는 이미 올라가 있어야 한다.

| 단계 | 모듈 | 펌웨어 | 연결할 포트 |
|---|---|---|---|
| 사진 | 팬·틸트 리그 + 카메라 | `../sunlight_scanner/rig_motor`, `xiao_cam_serial` | 리그 아두이노, XIAO 카메라 (서로 다른 COM 포트) |
| 표면 | 표면 온습도 기기 | `../mold_scanner/surface_scan` | 아두이노 1개 |

1. **방 정보**: 방 이름, 장소, 위도·경도, 북쪽 방위각, 카메라 높이 입력 (이름 외에는 비워도 됨).
   위도·경도가 있으면 표면 스캔 직후 기상청 초단기실황을 받아 `outdoor.csv`에 저장한다.
2. **사진**: 리그·카메라 포트를 고르고 Connect. 미리보기를 보며 W/A/S/D(또는 버튼)로 수평을 맞추고,
   1/2/3으로 이동 단위(0.35°/2°/10°)를 바꾼다. Enter -> 자동 촬영(약 21장). 사진 없이 진행하려면 Skip photos
   (그러면 일조 계산은 불가).
3. **표면 스캔**: 사진 모듈을 빼고 표면 모듈을 연결한다. 기기를 **먼저** 사진 모듈이 있던 자리에 수평으로 두고,
   센서가 **사진 촬영을 시작할 때 카메라가 보던 방향(수평 맞추기를 마친 yaw 0°)** 을 보게 한 뒤 포트와 벽 재질을 고르고 Start
   — 포트가 열리면 우노가 리셋되며 그 자세를 az 0°, el 0°로 잡는다. 20면, 약 1분 30초.
   앱은 이 시작 자세가 사진 모듈의 시작 자세(수평, 0°)와 같다고 가정하고 곰팡이 색을 방 모델에 칠한다.
4. **Create room**: 결과 화면(일조·곰팡이·침수)으로 이동한다.

- 장치 관리자(또는 Arduino IDE)에서 각 보드의 COM 번호를 확인한다. 포트 목록에 없으면 Refresh ports.
- 다른 프로그램(Arduino IDE 시리얼 모니터 등)이 포트를 열고 있으면 연결이 안 된다.
- 원본 데이터는 `scans/<id>/`에 남는다 (`photos/`, `surface/`, 다시 업로드할 수 있는 `bundle.zip`).
- 북쪽(방위각): 입력칸을 비워 두면 **사진 모듈의 방위 센서(GY-BNO055)** 로 잰 값을 쓴다 (`panorama.py`와 같은 방식).
  리그를 연결(Connect)하기 **전에** 카메라가 센서 X축 방향을 보게 두면, 연결 직후 모터가 움직이기 전에 북쪽을 잰다.
  수평 맞추기에서 돌린 팬 각도까지 반영해 `photos/meta.json`의 `north`에 저장하고, 방의 방위각으로 쓴다.
  직접 입력한 값이 있으면 그 값이 우선하고, 센서가 없으면 표면 모듈의 값, 그것도 없으면 0°로 계산된다.
  결과 화면 위쪽 막대에서 나중에 고칠 수 있다.
- zip 번들도 `photos/meta.json`에 `north`(`panorama.py`가 만든 것)가 있으면 방위각을 자동으로 읽는다.

### 3. 이미 있는 번들 올리기 (Home -> Upload Scan)
`.zip` 번들을 끌어다 놓는다. 테스트용으로 폴더에 있는 `bundle.zip`을 써도 된다.

### 추가 설정 (선택)
- 일조의 3D 방 모델 단계: `py -m pip install torch transformers` + 기존 스크립트가 쓰는 HorizonNet 가중치.
- 침수 탭 장소 검색: 서버가 `../mold_scanner/kma.py`의 `KAKAO_REST_KEY`로 대신 검색한다 (`/api/places`, JavaScript 키 불필요).
- 곰팡이 시뮬레이션의 날씨 데이터: `../mold_scanner/kma.py`의 기상청 인증키.

## Notes

- Home: upload a scan bundle (.zip). Format: see `rs/bundle.py`. Make a test bundle:
  `python tools/make_test_zip.py "../mold_scanner/data/20261021_2225" test_bundle.zip --heading 30 --fake-panorama`
- Existing scripts in ../sunlight_scanner, ../mold_scanner and ../flood_simulator are used unchanged
  (the Flood tab uses an English copy in static/flood).
- The Sunlight room-model step needs torch + transformers + the HorizonNet weights used by the existing script.
- Scan wizard (Home -> Start Room Scan): runs the two modules one after the other, using the classes in
  ../sunlight_scanner/panorama.py (rig_motor + XIAO camera) and ../mold_scanner/logger.py (surface_scan).
  Raw data and a re-uploadable bundle.zip are kept in scans/<id>/.
- Settings: config.yaml. API keys (KMA, Kakao REST) are read from ../mold_scanner/kma.py.

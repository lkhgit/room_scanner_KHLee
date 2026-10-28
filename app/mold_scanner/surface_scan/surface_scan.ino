/*
 * 표면 온습도 측정 기기 — 정이십면체 20면 IR 스캔 (데이터 수집용 펌웨어 v2)
 *
 * 하드웨어
 *   아랫단(팬) 28BYJ-48 + ULN2003 : IN1..IN4 -> D11, D10, D9, D8
 *   윗단(틸트) 28BYJ-48 + ULN2003 : IN1..IN4 -> D7, D6, D5, D4  (※ IN4를 D4로 가정)
 *   GY-906(MLX90614, 0x5A), SHT41(0x44), [선택] BNO055(0x28 또는 0x29)
 *   I2C 공유: SDA->A4, SCL->A5
 *
 * 좌표계
 *   전원 투입 시 자세(센서 정면·수평)를 방위각 0°, 고도각 0°로 둔다.
 *   정이십면체는 꼭짓점 하나가 천정을 향한다.
 *   위 5면 +52.62°, 중간 10면 ±10.81°, 아래 5면 -52.62°, 방위각 36° 간격 10열.
 *
 * 시리얼 프로토콜 (115200bps, 한 줄 단위)
 *   '#'로 시작하는 줄은 메타 정보, 그 외는 데이터 행이다.
 *   #HDR,<열 이름들>                 데이터 열 정의 (시작 시 1회)
 *   #SESSION,heading_deg,<값>,mag_calib,<0-3>,sys_calib,<0-3>
 *                                     기기 방위각 (BNO055 사용 시, 모터 통전 전에 측정)
 *   #SCAN_BEGIN,<scan_id>
 *   <scan_id>,,<face>,<zone>,<az>,<el>,<t_surf>,<t_surf_sd>,<t_air>,<rh>,<t_sensor>   x 20
 *   #SCAN_END,<scan_id>
 *   #ABORT,<scan_id>                  스캔 중단 시 (데이터 행 없음)
 *   #INFO,<메시지>                     사람이 읽는 안내
 *   데이터 행의 timestamp 열은 비워 두며, PC 로거가 채운다.
 *
 * 시리얼 명령
 *   전원을 켜면 모터를 움직이지 않고 대기한다 (#INFO,ready 출력).
 *   n : 스캔 1회 (끝나면 원위치로 돌아와 다시 대기)
 *   s : 스캔 중 즉시 정지 -> 원위치 -> 대기
 *   반복 측정이 필요하면 PC 로거가 정해진 간격으로 n을 보낸다.
 *
 * 백래시 보정: 모든 목표 위치에 +방향으로만 접근 (틸트 +는 위쪽).
 * 처짐 방지: 스캔 중 틸트 코일 전류를 계속 유지하고, 스캔 시작마다 틸트 0°를 다시 맞춘다.
 * 측정: 이동 후 2 s 대기, 약 1 s 동안 10회 읽어 평균. 스캔 1회 약 1분 반.
 *
 * 라이브러리: Adafruit MLX90614, Adafruit SHT4x, [선택] Adafruit BNO055
 */

#include <Wire.h>
#include <math.h>
#include <Adafruit_MLX90614.h>
#include <Adafruit_SHT4x.h>

#define USE_BNO055 0                   // BNO055를 달았으면 1로 바꾼다

#if USE_BNO055
#include <Adafruit_BNO055.h>
#endif

// ---------------- 설정 ----------------
const uint8_t PAN_PINS[4]  = {11, 10, 9, 8};
const uint8_t TILT_PINS[4] = {7, 6, 5, 4};

const float STEPS_PER_REV = 4096.0;    // 28BYJ-48 하프스텝 기준(실측 4076 근처일 수 있음)
const int   PAN_DIR  = 1;              // 회전 방향이 반대면 -1
const int   TILT_DIR = 1;              // +각도가 위를 보지 않으면 -1
const long  BACKLASH_OVERSHOOT = 80;   // 백래시보다 충분히 큰 값(스텝)
const unsigned int  STEP_US   = 1200;  // 스텝 간격(us)
// IR 측정 시간: MLX90614 기본 필터의 안정 시간은 약 0.85 s (Melexis 필터 앱노트).
// 그 2배 이상을 기다린 뒤 약 1 s에 걸쳐 10회 읽어 평균한다.
const unsigned long SETTLE_MS = 2000;  // 이동 후 대기 (진동 감쇠 + 필터 안정)
const int   N_SAMPLES = 10;            // IR 평균 횟수
const unsigned long SAMPLE_INTERVAL_MS = 100; // 읽기 간격
// 틸트 처짐 방지: 스캔 중에는 틸트 코일 전류를 유지한다.
// 원위치에서 대기할 때도 유지하려면 true (모터가 계속 따뜻해지고, IR 센서에 열이 전해질 수 있음).
const bool  HOLD_TILT_WHEN_IDLE = false;
const bool  COL0_UPPER = true;         // 방위각 0° 열이 위쪽 2면이면 true

#if USE_BNO055
const uint8_t BNO_ADDR = 0x28;         // I2C 스캐너 결과가 0x29면 바꾼다
const unsigned long BNO_CALIB_TIMEOUT_MS = 60000UL; // 지자기 보정 대기 최대 시간
#endif

// ---------------- 센서 ----------------
Adafruit_MLX90614 mlx;
Adafruit_SHT4x    sht4;
#if USE_BNO055
Adafruit_BNO055 bno(55, BNO_ADDR);
#endif

// ---------------- 상태 ----------------
volatile bool stopReq = false;
bool aligned = false;                  // 백래시 정렬을 했는지
bool scanning = false;
bool scanNowReq = false;
unsigned long scanCount = 0;           // scan_id (전원을 켤 때마다 1부터)

void pollSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if ((c == 's' || c == 'S') && scanning) stopReq = true;
    else if ((c == 'n' || c == 'N') && !scanning) scanNowReq = true;  // 스캔 중의 n은 무시
  }
}

// ---------------- 스테퍼 ----------------
const uint8_t SEQ[8][4] = {
  {1,0,0,0},{1,1,0,0},{0,1,0,0},{0,1,1,0},
  {0,0,1,0},{0,0,1,1},{0,0,0,1},{1,0,0,1}
};

struct Stepper28 {
  const uint8_t* pins;
  int    dir;
  long   pos;
  int8_t phase;

  void begin() {
    for (int k = 0; k < 4; k++) pinMode(pins[k], OUTPUT);
    pos = 0; phase = 0;
    release();
  }
  void writePhase() {
    for (int k = 0; k < 4; k++) digitalWrite(pins[k], SEQ[phase][k]);
  }
  void stepOnce(int s) {
    phase = (phase + s * dir + 8) % 8;
    writePhase();
    pos += s;
    delayMicroseconds(STEP_US);
  }
  void moveRaw(long target) {
    while (pos != target) {
      stepOnce(target > pos ? 1 : -1);
      pollSerial();
    }
  }
  void moveTo(long target) {           // 항상 +방향으로 도달 (백래시 제거)
    if (target < pos) moveRaw(target - BACKLASH_OVERSHOOT);
    moveRaw(target);
  }
  void moveToDeg(float deg) {
    moveTo(lround(deg / 360.0 * STEPS_PER_REV));
  }
  void takeUpBacklash() {
    writePhase();
    delay(5);
    moveRaw(-BACKLASH_OVERSHOOT);
    moveRaw(0);
  }
  void release() {
    for (int k = 0; k < 4; k++) digitalWrite(pins[k], LOW);
  }
};

Stepper28 pan  = {PAN_PINS,  PAN_DIR,  0, 0};
Stepper28 tilt = {TILT_PINS, TILT_DIR, 0, 0};

// ---------------- 면 목록 ----------------
const int N_FACES = 20;
float faceAz[N_FACES], faceEl[N_FACES];
char  faceZone[N_FACES];
float faceT[N_FACES], faceSD[N_FACES], faceTs[N_FACES];

void buildFaces() {
  const float E_MID = atan((3.0 - sqrt(5.0)) / 4.0) * 180.0 / PI;  // 10.812°
  const float E_CAP = atan((3.0 + sqrt(5.0)) / 4.0) * 180.0 / PI;  // 52.623°
  int n = 0;
  for (int j = 0; j < 10; j++) {
    float az = 36.0 * j;
    bool upper = (j % 2 == 0) ? COL0_UPPER : !COL0_UPPER;
    if (upper) {                       // 아래 -> 위 순서
      faceAz[n] = az; faceEl[n] =  E_MID; faceZone[n++] = 'M';
      faceAz[n] = az; faceEl[n] =  E_CAP; faceZone[n++] = 'T';
    } else {
      faceAz[n] = az; faceEl[n] = -E_CAP; faceZone[n++] = 'B';
      faceAz[n] = az; faceEl[n] = -E_MID; faceZone[n++] = 'M';
    }
  }
}

// ---------------- 측정 ----------------
// IR 물체 온도 평균·표준편차와 센서 자체 온도 평균
void readIR(float &mean, float &sd, float &ts) {
  float s = 0, s2 = 0, sa = 0;
  for (int i = 0; i < N_SAMPLES; i++) {
    float v = mlx.readObjectTempC();
    s += v; s2 += v * v;
    sa += mlx.readAmbientTempC();
    delay(SAMPLE_INTERVAL_MS);
    pollSerial();
  }
  mean = s / N_SAMPLES;
  float var = s2 / N_SAMPLES - mean * mean;
  sd   = sqrt(var > 0 ? var : 0);
  ts   = sa / N_SAMPLES;
}

void goHome() {
  tilt.moveToDeg(0);                   // 틸트는 전류를 유지한 채 이동
  pan.moveToDeg(0);                    // 역방향으로 돌아와 케이블 꼬임 방지
  pan.release();
  if (!HOLD_TILT_WHEN_IDLE) tilt.release();
}

void printHeader() {
  Serial.println(F("#HDR,scan_id,timestamp,face,zone,az_deg,el_deg,t_surf_c,t_surf_sd_c,t_air_c,rh_pct,t_sensor_c"));
}

void runScan() {
  if (!aligned) {                      // 첫 스캔 직전에 한 번만 팬 기어 유격 정렬
    pan.takeUpBacklash();
    pan.release();
    aligned = true;
  }
  // 틸트는 대기 중 전류가 꺼져 처졌을 수 있으므로 스캔마다 아래에서 0°로 다시 올려 맞춘다.
  // 이후 스캔이 끝날 때까지 틸트 전류를 유지한다.
  tilt.takeUpBacklash();
  scanning = true;
  stopReq = false;
  scanCount++;
  Serial.print(F("#SCAN_BEGIN,")); Serial.println(scanCount);

  for (int i = 0; i < N_FACES; i++) {
    pan.moveToDeg(faceAz[i]);
    if (stopReq) break;
    tilt.moveToDeg(faceEl[i]);
    if (stopReq) break;
    pan.release();                     // 팬(수직축)은 중력 부하가 없어 전류를 끈다. 틸트는 유지
    delay(SETTLE_MS);
    readIR(faceT[i], faceSD[i], faceTs[i]);
    if (stopReq) break;
  }

  if (stopReq) {
    stopReq = false;
    Serial.print(F("#ABORT,")); Serial.println(scanCount);
    goHome();
    scanning = false;
    Serial.println(F("#INFO,stopped at home. send n to scan again"));
    return;
  }

  goHome();

  sensors_event_t hum, temp;
  sht4.getEvent(&hum, &temp);
  float Ta = temp.temperature;
  float RH = hum.relative_humidity;

  for (int i = 0; i < N_FACES; i++) {
    Serial.print(scanCount);          Serial.print(F(",,"));
    Serial.print(i + 1);              Serial.print(',');
    Serial.print(faceZone[i]);        Serial.print(',');
    Serial.print(faceAz[i], 1);       Serial.print(',');
    Serial.print(faceEl[i], 2);       Serial.print(',');
    Serial.print(faceT[i], 2);        Serial.print(',');
    Serial.print(faceSD[i], 3);       Serial.print(',');
    Serial.print(Ta, 2);              Serial.print(',');
    Serial.print(RH, 1);              Serial.print(',');
    Serial.println(faceTs[i], 2);
  }
  Serial.print(F("#SCAN_END,")); Serial.println(scanCount);
  scanning = false;
}

#if USE_BNO055
// 모터 통전 전에 방위각을 잰다. 지자기 보정이 3이 될 때까지(최대 TIMEOUT) 기다린 뒤 20회 원형 평균.
void measureHeading() {
  if (!bno.begin()) {
    Serial.println(F("#INFO,BNO055 not found. check address"));
    Serial.println(F("#SESSION,heading_deg,NaN,mag_calib,0,sys_calib,0"));
    return;
  }
  bno.setExtCrystalUse(true);
  Serial.println(F("#INFO,BNO055 calibrating. rotate the whole device slowly, then place it at home pose"));
  uint8_t sys = 0, gyr = 0, acc = 0, mag = 0;
  unsigned long t0 = millis();
  while (millis() - t0 < BNO_CALIB_TIMEOUT_MS) {
    bno.getCalibration(&sys, &gyr, &acc, &mag);
    if (mag == 3) break;
    delay(200);
  }
  delay(2000);                         // 내려놓은 뒤 안정 대기
  float sx = 0, cy = 0;
  for (int i = 0; i < 20; i++) {
    sensors_event_t ev;
    bno.getEvent(&ev);
    float h = ev.orientation.x * PI / 180.0;
    sx += sin(h); cy += cos(h);
    delay(50);
  }
  float heading = atan2(sx, cy) * 180.0 / PI;
  if (heading < 0) heading += 360.0;
  bno.getCalibration(&sys, &gyr, &acc, &mag);
  Serial.print(F("#SESSION,heading_deg,")); Serial.print(heading, 1);
  Serial.print(F(",mag_calib,")); Serial.print(mag);
  Serial.print(F(",sys_calib,")); Serial.println(sys);
}
#endif

void setup() {
  Serial.begin(115200);
  Wire.begin();

  pan.begin();                         // 핀만 설정, 코일 전류 없음
  tilt.begin();

  if (!mlx.begin()) { Serial.println(F("#INFO,ERROR MLX90614 not found")); while (1); }
  if (!sht4.begin()) { Serial.println(F("#INFO,ERROR SHT41 not found"));   while (1); }
  sht4.setPrecision(SHT4X_HIGH_PRECISION);
  sht4.setHeater(SHT4X_NO_HEATER);

  buildFaces();
  printHeader();

#if USE_BNO055
  measureHeading();                    // 모터 통전 전에 측정
#endif

  // 모터는 움직이지 않고 n을 기다린다
  Serial.println(F("#INFO,ready"));
}

void loop() {
  pollSerial();
  if (scanNowReq) {
    scanNowReq = false;
    runScan();
  }
}

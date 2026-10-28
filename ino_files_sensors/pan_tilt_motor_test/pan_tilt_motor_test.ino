// ============================================================
//  Room Scanner 팬·틸트 리그 — 모터 키보드 테스트
//  28BYJ-48 (5V) + ULN2003 x2, Arduino UNO
//
//  PAN  (아랫단, 360°) : IN1~IN4 -> D7, D6, D5, D4
//  TILT (윗단,  180°) : IN1~IN4 -> D8, D9, D10, D11
//
//  시리얼 모니터 115200 baud. 키를 입력하고 Enter.
//  여러 글자를 한 번에 보내면 차례로 실행됨 (예: "ddd" = 3칸)
// ============================================================

// ---- 기본 설정 ----------------------------------------------
const long  STEPS_PER_REV   = 4096;          // 하프스텝 기준 (기어비 근사값)
const float STEPS_PER_DEG   = STEPS_PER_REV / 360.0;
const unsigned long STEP_INTERVAL_US = 1200; // 스텝 간격. 탈조되면 늘릴 것 (1200~2000)

const int8_t PAN_DIR  = 1;   // 방향이 반대로 돌면 -1로 변경
const int8_t TILT_DIR = 1;   // 방향이 반대로 돌면 -1로 변경

const float TILT_MIN_DEG = -90.0;  // 시작 위치(=0°) 기준 틸트 한계
const float TILT_MAX_DEG =  90.0;

// 하프스텝 시퀀스 (IN1, IN2, IN3, IN4)
const uint8_t HALF_SEQ[8][4] = {
  {1,0,0,0}, {1,1,0,0}, {0,1,0,0}, {0,1,1,0},
  {0,0,1,0}, {0,0,1,1}, {0,0,0,1}, {1,0,0,1}
};

// ---- 모터 구조체 --------------------------------------------
struct Motor {
  const char* name;
  uint8_t pins[4];
  int8_t  dirSign;
  long    pos;          // 현재 위치 (하프스텝)
  long    target;       // 목표 위치 (하프스텝)
  int8_t  phase;        // 0~7
  unsigned long lastStepUs;
  bool    hasLimit;
  long    minPos, maxPos;
  bool    holdAfterMove; // 정지 후 코일 유지 여부
  bool    moving;
};

Motor pan  = {"PAN",  {7, 6, 5, 4},   PAN_DIR,  0, 0, 0, 0, false, 0, 0, false, false};
Motor tilt = {"TILT", {8, 9, 10, 11}, TILT_DIR, 0, 0, 0, 0, true,  0, 0, true,  false};

float stepDeg = 5.0;  // 키 한 번당 회전 각도

// ---- 저수준 제어 --------------------------------------------
void writeCoils(Motor &m) {
  for (int i = 0; i < 4; i++) digitalWrite(m.pins[i], HALF_SEQ[m.phase][i]);
}

void releaseCoils(Motor &m) {
  for (int i = 0; i < 4; i++) digitalWrite(m.pins[i], LOW);
}

void stepOnce(Motor &m, int dir) {          // dir: +1 / -1 (논리 방향)
  int physical = dir * m.dirSign;
  m.phase = (m.phase + physical + 8) % 8;
  writeCoils(m);
  m.pos += dir;
}

float toDeg(long steps) { return steps / STEPS_PER_DEG; }

void printStatus() {
  Serial.print(F("PAN: "));   Serial.print(toDeg(pan.pos), 2);
  Serial.print(F("°  |  TILT: ")); Serial.print(toDeg(tilt.pos), 2);
  Serial.print(F("°  |  step size: ")); Serial.print(stepDeg, 1);
  Serial.println(F("°"));
}

// 비차단(non-blocking) 구동: loop()마다 호출
void updateMotor(Motor &m) {
  if (m.pos != m.target) {
    unsigned long now = micros();
    if (now - m.lastStepUs >= STEP_INTERVAL_US) {
      m.lastStepUs = now;
      stepOnce(m, (m.target > m.pos) ? 1 : -1);
      m.moving = true;
    }
  } else if (m.moving) {
    m.moving = false;
    if (!m.holdAfterMove) releaseCoils(m);
    Serial.print(m.name); Serial.print(F(" 도착 -> "));
    printStatus();
  }
}

// ---- 명령 ---------------------------------------------------
void moveBy(Motor &m, float deg) {
  long t = m.target + lround(deg * STEPS_PER_DEG);
  if (m.hasLimit) {
    if (t > m.maxPos) { t = m.maxPos; Serial.print(m.name); Serial.println(F(": 상한 도달")); }
    if (t < m.minPos) { t = m.minPos; Serial.print(m.name); Serial.println(F(": 하한 도달")); }
  }
  m.target = t;
}

void setZero(Motor &m) {
  m.pos = m.target = 0;
  if (m.hasLimit) {
    m.minPos = lround(TILT_MIN_DEG * STEPS_PER_DEG);
    m.maxPos = lround(TILT_MAX_DEG * STEPS_PER_DEG);
  }
}

void printHelp() {
  Serial.println(F("\n==== 팬·틸트 모터 테스트 ===="));
  Serial.println(F(" a / d : PAN  -/+ 회전"));
  Serial.println(F(" w / s : TILT +/- 회전"));
  Serial.println(F(" 1 2 3 4 : 스텝 크기 1° / 5° / 15° / 45°"));
  Serial.println(F(" h : 두 모터 모두 0°로 복귀"));
  Serial.println(F(" z : 현재 위치를 0°로 설정 (정지 상태에서)"));
  Serial.println(F(" x : 즉시 정지"));
  Serial.println(F(" t : TILT 코일 유지 on/off (off면 카메라 무게로 처질 수 있음)"));
  Serial.println(F(" r : 두 모터 코일 해제 (발열 방지)"));
  Serial.println(F(" p : 현재 위치 출력   ? : 도움말"));
  Serial.println(F(" * 틸트는 시작 위치를 0°(수평)로 보고 ±90°로 제한됨\n"));
}

void handleKey(char c) {
  switch (c) {
    case 'a': moveBy(pan, -stepDeg); break;
    case 'd': moveBy(pan, +stepDeg); break;
    case 'w': moveBy(tilt, +stepDeg); break;
    case 's': moveBy(tilt, -stepDeg); break;

    case '1': stepDeg = 1;  printStatus(); break;
    case '2': stepDeg = 5;  printStatus(); break;
    case '3': stepDeg = 15; printStatus(); break;
    case '4': stepDeg = 45; printStatus(); break;

    case 'h': pan.target = 0; tilt.target = 0; break;

    case 'z':
      if (pan.pos == pan.target && tilt.pos == tilt.target) {
        setZero(pan); setZero(tilt);
        Serial.println(F("현재 위치를 0°로 설정"));
      } else {
        Serial.println(F("정지 상태에서만 가능 (x로 정지 후 다시)"));
      }
      break;

    case 'x':
      pan.target = pan.pos; tilt.target = tilt.pos;
      Serial.println(F("정지")); printStatus();
      break;

    case 't':
      tilt.holdAfterMove = !tilt.holdAfterMove;
      if (!tilt.holdAfterMove && !tilt.moving) releaseCoils(tilt);
      if (tilt.holdAfterMove) writeCoils(tilt);
      Serial.print(F("TILT 코일 유지: "));
      Serial.println(tilt.holdAfterMove ? F("ON") : F("OFF"));
      break;

    case 'r':
      pan.target = pan.pos; tilt.target = tilt.pos;
      releaseCoils(pan); releaseCoils(tilt);
      tilt.holdAfterMove = false;
      Serial.println(F("코일 해제 (TILT 유지도 OFF)"));
      break;

    case 'p': printStatus(); break;
    case '?': printHelp(); break;

    case '\r': case '\n': case ' ': break;
    default:
      Serial.print(F("알 수 없는 키: ")); Serial.println(c);
  }
}

// ---- setup / loop -------------------------------------------
void setup() {
  for (int i = 0; i < 4; i++) {
    pinMode(pan.pins[i], OUTPUT);
    pinMode(tilt.pins[i], OUTPUT);
  }
  releaseCoils(pan);
  setZero(pan);
  setZero(tilt);
  writeCoils(tilt);   // 틸트는 시작부터 위치 유지

  Serial.begin(115200);
  printHelp();
  printStatus();
}

void loop() {
  while (Serial.available()) handleKey((char)Serial.read());
  updateMotor(pan);
  updateMotor(tilt);
}

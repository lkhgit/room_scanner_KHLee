// ============================================================
//  팬·틸트 리그 모터 제어 (Arduino UNO + ULN2003 x2 + 28BYJ-48 x2)
//  - 아랫단(팬)  IN1~IN4 -> D8, D9, D10, D11
//  - 윗단(틸트) IN1~IN4 -> D4, D5, D6, D7
//
//  시리얼 명령 (115200 baud, 줄바꿈 '\n'으로 끝):
//    P <n>   팬 모터를 n 하프스텝 상대 이동 (음수 = 역방향)
//    T <n>   틸트 모터를 n 하프스텝 상대 이동
//    S <us>  하프스텝 간 지연(µs) 설정 (800~10000)
//    H <0|1> 이동 후 코일 유지 여부 (0 = 이동 후 전류 차단, 기본값)
//    R       두 모터 모두 전류 차단
//    ?       핑 -> "READY"
//  이동이 끝나면 "OK", 잘못된 명령이면 "ERR"을 보냄.
// ============================================================

const uint8_t PAN_PINS[4]  = {8, 9, 10, 11};
const uint8_t TILT_PINS[4] = {4, 5, 6, 7};

// 하프스텝 시퀀스 (IN1, IN2, IN3, IN4)
const uint8_t SEQ[8][4] = {
  {1,0,0,0}, {1,1,0,0}, {0,1,0,0}, {0,1,1,0},
  {0,0,1,0}, {0,0,1,1}, {0,0,0,1}, {1,0,0,1}
};

int8_t panPhase = 0;
int8_t tiltPhase = 0;
unsigned int stepDelayUs = 1500;   // 정속 구간 하프스텝 간격
bool holdTorque = false;           // 발열을 줄이기 위해 기본은 이동 후 전류 차단

const long RAMP_STEPS = 120;       // 가감속 구간 길이 (하프스텝)
const unsigned int RAMP_EXTRA_US = 18;  // 가감속 구간에서 스텝당 추가 지연

String lineBuf;

void writeCoils(const uint8_t* pins, int8_t phase) {
  for (uint8_t i = 0; i < 4; i++) digitalWrite(pins[i], SEQ[phase][i]);
}

void releaseCoils(const uint8_t* pins) {
  for (uint8_t i = 0; i < 4; i++) digitalWrite(pins[i], LOW);
}

void moveMotor(const uint8_t* pins, int8_t &phase, long steps) {
  if (steps == 0) return;
  int8_t dir = (steps > 0) ? 1 : -1;
  long n = labs(steps);

  writeCoils(pins, phase);   // 현재 상(phase)을 먼저 잡아서 첫 스텝이 튀지 않게
  delay(5);

  for (long i = 0; i < n; i++) {
    phase = (phase + dir + 8) % 8;
    writeCoils(pins, phase);

    // 시작·끝 부분은 천천히 (탈조 방지)
    long fromEdge = min(i, n - 1 - i);
    unsigned int d = stepDelayUs;
    if (fromEdge < RAMP_STEPS) d += (unsigned int)(RAMP_STEPS - fromEdge) * RAMP_EXTRA_US;
    delayMicroseconds(d);
  }

  delay(20);
  if (!holdTorque) releaseCoils(pins);
}

void handleLine(String line) {
  line.trim();
  if (line.length() == 0) return;

  char cmd = line.charAt(0);
  long val = line.substring(1).toInt();

  switch (cmd) {
    case 'P': moveMotor(PAN_PINS, panPhase, val);   Serial.println("OK"); break;
    case 'T': moveMotor(TILT_PINS, tiltPhase, val); Serial.println("OK"); break;
    case 'S': stepDelayUs = constrain(val, 800, 10000); Serial.println("OK"); break;
    case 'H':
      holdTorque = (val != 0);
      if (!holdTorque) { releaseCoils(PAN_PINS); releaseCoils(TILT_PINS); }
      Serial.println("OK");
      break;
    case 'R': releaseCoils(PAN_PINS); releaseCoils(TILT_PINS); Serial.println("OK"); break;
    case '?': Serial.println("READY"); break;
    default:  Serial.println("ERR"); break;
  }
}

void setup() {
  for (uint8_t i = 0; i < 4; i++) {
    pinMode(PAN_PINS[i], OUTPUT);
    pinMode(TILT_PINS[i], OUTPUT);
  }
  releaseCoils(PAN_PINS);
  releaseCoils(TILT_PINS);

  Serial.begin(115200);
  Serial.println("READY");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') { handleLine(lineBuf); lineBuf = ""; }
    else if (c != '\r') lineBuf += c;
  }
}

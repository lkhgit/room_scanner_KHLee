// ============================================================
//  팬·틸트 리그 모터 제어 + 방위 센서 (Arduino UNO + ULN2003 x2 + 28BYJ-48 x2 + GY-BNO055)
//  - 아랫단(팬)  IN1~IN4 -> D8, D9, D10, D11
//  - 윗단(틸트) IN1~IN4 -> D4, D5, D6, D7
//  - GY-BNO055   VIN -> 5V, GND -> GND, SDA -> A4, SCL -> A5   (I2C 주소 0x29, 없으면 0x28 탐색)
//    라이브러리 설치 필요 없음 (Wire, EEPROM만 사용)
//    부팅할 때 bno055_compass.ino(ino_files_sensors/compass_sensor_calibration)가 EEPROM에 저장한 지자기 보정값을 불러와 센서에 넣는다.
//      (EEPROM 0~3: 확인용 값 0xB0055C01, 4~25: 보정 레지스터 0x55~0x6A 22바이트)
//    이 스케치는 EEPROM에 아무것도 쓰지 않으므로, 다른 스케치를 올려도 보정값은 그대로 남는다.
//
//  시리얼 명령 (115200 baud, 줄바꿈 '\n'으로 끝):
//    P <n>   팬 모터를 n 하프스텝 상대 이동 (음수 = 역방향)
//    T <n>   틸트 모터를 n 하프스텝 상대 이동
//    S <us>  하프스텝 간 지연(µs) 설정 (800~10000)
//    H <0|1> 이동 후 코일 유지 여부 (0 = 이동 후 전류 차단, 기본값)
//    R       두 모터 모두 전류 차단
//    N       방위 센서 값 1회 -> "N mx my mz ax ay az eh cal"
//              mx~mz: 자기장 (1/16 µT), ax~az: 가속도 (1/100 m/s², 가만히 있으면 위쪽을 가리킴)
//              eh: 센서 자체 heading (1/16°, 자북 기준)
//              cal: 보정 상태 바이트 (sys<<6 | gyro<<4 | accel<<2 | mag, 각 0~3)
//            센서가 없으면 "ERR imu"
//    I       센서 상태 -> "I <주소> <보정값 불러옴 0|1>"  (센서가 없으면 "ERR imu")
//    ?       핑 -> "READY"
//  이동이 끝나면 "OK", 잘못된 명령이면 "ERR"을 보냄.
// ============================================================
#include <Wire.h>
#include <EEPROM.h>

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

// ---------------- BNO055 ----------------
const uint8_t BNO_CHIP_ID   = 0x00;   // 0xA0이어야 함
const uint8_t BNO_PAGE_ID   = 0x07;
const uint8_t BNO_ACC_X     = 0x08;   // 가속도 X,Y,Z (LSB 먼저)
const uint8_t BNO_MAG_X     = 0x0E;   // 자기장 X,Y,Z
const uint8_t BNO_EUL_H     = 0x1A;   // Euler heading
const uint8_t BNO_OFFSETS   = 0x55;   // 보정 레지스터 시작 (22바이트)
const uint8_t BNO_CALIB     = 0x35;
const uint8_t BNO_UNIT_SEL  = 0x3B;
const uint8_t BNO_OPR_MODE  = 0x3D;
const uint8_t BNO_PWR_MODE  = 0x3E;
const uint8_t BNO_SYS_TRIG  = 0x3F;
const uint8_t MODE_CONFIG   = 0x00;
const uint8_t MODE_COMPASS  = 0x09;   // 가속도+지자기 (보정할 때와 같은 모드)

// bno055_compass.ino 와 같은 저장 형식
const uint32_t CALIB_MAGIC     = 0xB0055C01UL;
const int      CALIB_ADDR_MAGIC = 0;
const int      CALIB_ADDR_DATA  = 4;
const uint8_t  CALIB_LEN        = 22;

uint8_t bnoAddr = 0;                  // 0이면 센서 없음
bool calibLoaded = false;

bool bnoWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(bnoAddr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

bool bnoRead(uint8_t reg, uint8_t *buf, uint8_t n) {
  Wire.beginTransmission(bnoAddr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(bnoAddr, n) != n) return false;
  for (uint8_t i = 0; i < n; i++) buf[i] = Wire.read();
  return true;
}

bool bnoProbe(uint8_t addr) {
  bnoAddr = addr;
  uint8_t id = 0;
  for (uint8_t tries = 0; tries < 10; tries++) {     // 전원 직후에는 응답까지 시간이 걸림
    if (bnoRead(BNO_CHIP_ID, &id, 1) && id == 0xA0) return true;
    delay(100);
  }
  bnoAddr = 0;
  return false;
}

// EEPROM에 저장된 보정값을 센서에 넣음 (CONFIG 모드에서만 쓸 수 있음)
bool loadCalib() {
  uint32_t magic;
  EEPROM.get(CALIB_ADDR_MAGIC, magic);
  if (magic != CALIB_MAGIC) return false;
  for (uint8_t i = 0; i < CALIB_LEN; i++) {
    if (!bnoWrite(BNO_OFFSETS + i, EEPROM.read(CALIB_ADDR_DATA + i))) return false;
  }
  return true;
}

void bnoInit() {
  Wire.begin();
#if defined(WIRE_HAS_TIMEOUT)
  Wire.setWireTimeout(3000, true);                   // 센서가 빠져도 멈추지 않게
#endif
  if (!bnoProbe(0x29) && !bnoProbe(0x28)) return;
  bnoWrite(BNO_OPR_MODE, MODE_CONFIG); delay(25);
  bnoWrite(BNO_PAGE_ID, 0);
  bnoWrite(BNO_PWR_MODE, 0x00);       delay(10);     // normal
  bnoWrite(BNO_SYS_TRIG, 0x80);       delay(10);     // 외부 32.768kHz 크리스털 사용 (보정할 때와 동일)
  bnoWrite(BNO_UNIT_SEL, 0x80);       delay(10);     // m/s², 도(°)
  calibLoaded = loadCalib();          delay(10);
  bnoWrite(BNO_OPR_MODE, MODE_COMPASS); delay(30);
}

int16_t le16(const uint8_t *b) { return (int16_t)((uint16_t)b[0] | ((uint16_t)b[1] << 8)); }

void sendImu() {
  uint8_t mag[6], grv[6], eul[2], cal;
  if (!bnoAddr || !bnoRead(BNO_MAG_X, mag, 6) || !bnoRead(BNO_ACC_X, grv, 6) ||
      !bnoRead(BNO_EUL_H, eul, 2) || !bnoRead(BNO_CALIB, &cal, 1)) {
    Serial.println("ERR imu");
    return;
  }
  Serial.print("N");
  for (uint8_t i = 0; i < 6; i += 2) { Serial.print(' '); Serial.print(le16(mag + i)); }
  for (uint8_t i = 0; i < 6; i += 2) { Serial.print(' '); Serial.print(le16(grv + i)); }
  Serial.print(' '); Serial.print(le16(eul));
  Serial.print(' '); Serial.println(cal);
}

// ---------------- 모터 ----------------
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
    case 'N': sendImu(); break;
    case 'I':
      if (!bnoAddr) { Serial.println("ERR imu"); break; }
      Serial.print("I 0x"); Serial.print(bnoAddr, HEX);
      Serial.print(' ');    Serial.println(calibLoaded ? 1 : 0);
      break;
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
  bnoInit();
  Serial.println("READY");
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') { handleLine(lineBuf); lineBuf = ""; }
    else if (c != '\r') lineBuf += c;
  }
}

// GY-BNO055 자북 방위 + 지자기 보정값 EEPROM 저장/복원 (Arduino UNO)
// 필요 라이브러리: Adafruit BNO055, Adafruit Unified Sensor
//
// 처음 실행: 8자로 돌려 M=3이 되면 자동 저장
// 다음부터: 부팅 시 저장값 자동 복원
// 시리얼 명령(115200): 's' 지금 값 저장 / 'c' 저장값 삭제

#include <Wire.h>
#include <EEPROM.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>

const uint8_t BNO_ADDR = 0x29;
Adafruit_BNO055 bno(55, BNO_ADDR, &Wire);

const uint32_t MAGIC = 0xB0055C01;  // 저장된 데이터가 유효한지 표시
const int ADDR_MAGIC = 0;
const int ADDR_DATA = 4;
const uint8_t OFFSET_LEN = 22;       // 가속도·지자기·자이로 오프셋 + 반경
bool saved = false;

// 오프셋 레지스터(0x55~0x6A)를 직접 읽음 (CONFIG 모드에서만 읽기 가능)
bool readOffsets(uint8_t *buf) {
  bno.setMode(OPERATION_MODE_CONFIG);
  delay(25);
  Wire.beginTransmission(BNO_ADDR);
  Wire.write(0x55);
  bool ok = (Wire.endTransmission() == 0);
  if (ok) {
    uint8_t n = Wire.requestFrom(BNO_ADDR, OFFSET_LEN);
    for (uint8_t i = 0; i < n; i++) buf[i] = Wire.read();
    ok = (n == OFFSET_LEN);
  }
  bno.setMode(OPERATION_MODE_COMPASS);
  delay(20);
  return ok;
}

void printOffsets(const uint8_t *buf) {
  // 지자기 오프셋 X, Y, Z (리틀엔디안 int16)
  int16_t mx = buf[6] | (buf[7] << 8);
  int16_t my = buf[8] | (buf[9] << 8);
  int16_t mz = buf[10] | (buf[11] << 8);
  Serial.print("  Mag offset: ");
  Serial.print(mx); Serial.print(' ');
  Serial.print(my); Serial.print(' ');
  Serial.println(mz);
}

void saveOffsets() {
  uint8_t buf[OFFSET_LEN];
  if (!readOffsets(buf)) {
    Serial.println("! 오프셋 읽기 실패");
    return;
  }
  EEPROM.put(ADDR_MAGIC, MAGIC);
  for (uint8_t i = 0; i < OFFSET_LEN; i++) EEPROM.update(ADDR_DATA + i, buf[i]);
  Serial.println(">> 보정값 저장 완료");
  printOffsets(buf);
  saved = true;
}

void setup() {
  Serial.begin(115200);
  Serial.println("start");
  if (!bno.begin(OPERATION_MODE_COMPASS)) {
    Serial.println("BNO055 없음: 배선/주소 확인");
    while (1) {}
  }
  delay(1000);
  bno.setExtCrystalUse(true);

  uint32_t magic;
  EEPROM.get(ADDR_MAGIC, magic);
  if (magic == MAGIC) {
    uint8_t buf[OFFSET_LEN];
    for (uint8_t i = 0; i < OFFSET_LEN; i++) buf[i] = EEPROM.read(ADDR_DATA + i);
    bno.setSensorOffsets(buf);
    Serial.println(">> 저장된 보정값 복원");
    printOffsets(buf);
    saved = true;
  } else {
    Serial.println(">> 저장된 보정값 없음: 8자로 돌려 M=3을 만드세요");
  }
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 's') saveOffsets();
    if (c == 'c') {
      EEPROM.put(ADDR_MAGIC, (uint32_t)0);
      saved = false;
      Serial.println(">> 저장값 삭제. 다시 M=3이 되면 자동 저장됩니다");
    }
  }

  uint8_t sys, gyro, accel, mag;
  bno.getCalibration(&sys, &gyro, &accel, &mag);

  if (!saved && mag == 3) {
    Serial.println(">> M=3 도달, 자동 저장");
    saveOffsets();
  }

  sensors_event_t e;
  bno.getEvent(&e, Adafruit_BNO055::VECTOR_EULER);

  Serial.print("M=");
  Serial.print(mag);
  Serial.print("  heading=");
  Serial.println(e.orientation.x, 1);  // 0 = 자북, 시계방향 증가

  delay(200);
}

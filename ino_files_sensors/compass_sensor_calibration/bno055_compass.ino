// GY-BNO055 magnetic heading + saving/restoring the magnetometer calibration in EEPROM (Arduino UNO)
// Libraries needed: Adafruit BNO055, Adafruit Unified Sensor
//
// First run: move it in a figure eight; it saves automatically once M=3
// After that: the saved values are restored automatically at boot
// Serial commands (115200): 's' save the current values / 'c' delete the saved values

#include <Wire.h>
#include <EEPROM.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>

const uint8_t BNO_ADDR = 0x29;
Adafruit_BNO055 bno(55, BNO_ADDR, &Wire);

const uint32_t MAGIC = 0xB0055C01;  // marks the saved data as valid
const int ADDR_MAGIC = 0;
const int ADDR_DATA = 4;
const uint8_t OFFSET_LEN = 22;       // accelerometer, magnetometer and gyro offsets + radii
bool saved = false;

// Reads the offset registers (0x55-0x6A) directly (only readable in CONFIG mode).
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

// Prints the magnetometer offsets from a calibration buffer.
void printOffsets(const uint8_t *buf) {
  // magnetometer offsets X, Y, Z (little-endian int16)
  int16_t mx = buf[6] | (buf[7] << 8);
  int16_t my = buf[8] | (buf[9] << 8);
  int16_t mz = buf[10] | (buf[11] << 8);
  Serial.print("  Mag offset: ");
  Serial.print(mx); Serial.print(' ');
  Serial.print(my); Serial.print(' ');
  Serial.println(mz);
}

// Reads the current calibration and saves it to EEPROM.
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

// Starts the sensor in compass mode and restores the saved calibration if there is one.
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

// Handles the s/c commands, saves once M reaches 3, and prints the calibration state and heading.
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
  Serial.println(e.orientation.x, 1);  // 0 = magnetic north, increasing clockwise

  delay(200);
}

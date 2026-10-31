/*
 * cam_probe.ino — camera sensor diagnostics for the XIAO ESP32S3 Sense (checks directly, without the camera library)
 *
 * What it does
 *   1) feeds XCLK (10 MHz, 20 MHz) to the camera
 *   2) scans the SCCB (I2C) bus for addresses that answer
 *   3) reads the ID registers of known sensors several times and prints them
 *      - OV2640 : address 0x30, PID = 0x26 0x42
 *      - OV3660 : address 0x3C, PID = 0x36 0x60
 *      - OV5640 : address 0x3C, PID = 0x56 0x40
 *   Values that change every time mean a bad contact/signal; steady values that differ from the table mean another sensor.
 *
 * Board settings as for xiao_cam_serial.ino (XIAO_ESP32S3, USB CDC On Boot: Enabled)
 * After uploading, open the serial monitor (115200) and press the XIAO's R button to print the results.
 */
#include <Wire.h>

#define XCLK_PIN 10
#define SDA_PIN  40
#define SCL_PIN  39

// Reads an 8-bit register at an 8-bit address; negative on an I2C error.
int readReg8(uint8_t addr, uint8_t reg) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return -1;
  if (Wire.requestFrom(addr, (uint8_t)1) != 1) return -2;
  return Wire.read();
}

// Writes an 8-bit register at an 8-bit address.
bool writeReg8(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

// Reads an 8-bit register at a 16-bit address; negative on an I2C error.
int readReg16(uint8_t addr, uint16_t reg) {
  Wire.beginTransmission(addr);
  Wire.write(reg >> 8);
  Wire.write(reg & 0xFF);
  if (Wire.endTransmission(false) != 0) return -1;
  if (Wire.requestFrom(addr, (uint8_t)1) != 1) return -2;
  return Wire.read();
}

// Feeds XCLK at the given frequency, scans the bus and reads the sensor IDs five times.
void probe(uint32_t xclkHz) {
  Serial.printf("\n===== XCLK %lu MHz =====\n", xclkHz / 1000000);
  ledcDetach(XCLK_PIN);
  if (!ledcAttach(XCLK_PIN, xclkHz, 1)) {
    Serial.println("XCLK 설정 실패");
    return;
  }
  ledcWrite(XCLK_PIN, 1);   // 50% duty
  delay(100);

  Serial.print("I2C 응답 주소:");
  int found = 0;
  for (uint8_t a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) {
      Serial.printf(" 0x%02X", a);
      found++;
    }
  }
  Serial.println(found ? "" : " 없음 (센서가 전혀 응답하지 않음)");

  for (int i = 0; i < 5; i++) {
    // OV2640: 0xFF=0x01 (sensor bank), then 0x0A/0x0B
    int h = -9, l = -9;
    if (writeReg8(0x30, 0xFF, 0x01)) {
      h = readReg8(0x30, 0x0A);
      l = readReg8(0x30, 0x0B);
    }
    // OV3660/OV5640: 16-bit registers 0x300A/0x300B
    int h2 = readReg16(0x3C, 0x300A);
    int l2 = readReg16(0x3C, 0x300B);
    Serial.printf("  읽기 %d:  @0x30 PID=%02X %02X   |  @0x3C PID=%02X %02X\n",
                  i + 1, h & 0xFF, l & 0xFF, h2 & 0xFF, l2 & 0xFF);
    if (h < 0) Serial.println("     (0x30 응답 없음)");
    if (h2 < 0) Serial.println("     (0x3C 응답 없음)");
    delay(50);
  }
}

// Runs the probe at 10 MHz and 20 MHz once at start-up.
void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\n카메라 센서 진단 시작");
  Wire.begin(SDA_PIN, SCL_PIN, 100000);
  probe(10000000);
  probe(20000000);
  Serial.println("\n끝. 위 결과 전체를 복사해서 보내 주세요.");
}

// Nothing to do after setup().
void loop() {}

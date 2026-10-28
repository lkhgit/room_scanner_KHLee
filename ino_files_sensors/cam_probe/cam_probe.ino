/*
 * cam_probe.ino — XIAO ESP32S3 Sense 카메라 센서 진단용 (카메라 라이브러리 없이 직접 확인)
 *
 * 하는 일
 *   1) 카메라에 XCLK(10MHz, 20MHz)를 공급
 *   2) SCCB(I2C) 버스에서 응답하는 주소를 스캔
 *   3) 알려진 센서의 ID 레지스터를 여러 번 읽어 출력
 *      - OV2640 : 주소 0x30, PID = 0x26 0x42
 *      - OV3660 : 주소 0x3C, PID = 0x36 0x60
 *      - OV5640 : 주소 0x3C, PID = 0x56 0x40
 *   값이 매번 다르게 나오면 접촉/신호 불량, 일정한데 표와 다르면 다른 센서.
 *
 * 보드 설정은 xiao_cam_serial.ino와 동일 (XIAO_ESP32S3, USB CDC On Boot: Enabled)
 * 업로드 후 시리얼 모니터(115200)를 열고 XIAO의 R 버튼을 누르면 결과가 출력됨.
 */
#include <Wire.h>

#define XCLK_PIN 10
#define SDA_PIN  40
#define SCL_PIN  39

int readReg8(uint8_t addr, uint8_t reg) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return -1;
  if (Wire.requestFrom(addr, (uint8_t)1) != 1) return -2;
  return Wire.read();
}

bool writeReg8(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

int readReg16(uint8_t addr, uint16_t reg) {
  Wire.beginTransmission(addr);
  Wire.write(reg >> 8);
  Wire.write(reg & 0xFF);
  if (Wire.endTransmission(false) != 0) return -1;
  if (Wire.requestFrom(addr, (uint8_t)1) != 1) return -2;
  return Wire.read();
}

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
    // OV2640: 0xFF=0x01 (sensor bank) 후 0x0A/0x0B
    int h = -9, l = -9;
    if (writeReg8(0x30, 0xFF, 0x01)) {
      h = readReg8(0x30, 0x0A);
      l = readReg8(0x30, 0x0B);
    }
    // OV3660/OV5640: 16비트 레지스터 0x300A/0x300B
    int h2 = readReg16(0x3C, 0x300A);
    int l2 = readReg16(0x3C, 0x300B);
    Serial.printf("  읽기 %d:  @0x30 PID=%02X %02X   |  @0x3C PID=%02X %02X\n",
                  i + 1, h & 0xFF, l & 0xFF, h2 & 0xFF, l2 & 0xFF);
    if (h < 0) Serial.println("     (0x30 응답 없음)");
    if (h2 < 0) Serial.println("     (0x3C 응답 없음)");
    delay(50);
  }
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("\n카메라 센서 진단 시작");
  Wire.begin(SDA_PIN, SCL_PIN, 100000);
  probe(10000000);
  probe(20000000);
  Serial.println("\n끝. 위 결과 전체를 복사해서 보내 주세요.");
}

void loop() {}

// I2C 스캐너: 버스에 연결된 장치 주소를 모두 출력
#include <Wire.h>

void setup() {
  Serial.begin(115200);
  Wire.begin();
  delay(1000);  // BNO055 부팅 대기 (약 650ms 필요)
  Serial.println("I2C 스캔 시작");

  int found = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.print("발견: 0x");
      if (addr < 16) Serial.print('0');
      Serial.println(addr, HEX);
      found++;
    }
  }
  if (found == 0) Serial.println("장치 없음");
  Serial.println("스캔 끝");
}

void loop() {}

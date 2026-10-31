// I2C scanner: prints the address of every device on the bus
#include <Wire.h>

// Scans all I2C addresses once at start-up and prints the ones that answer.
void setup() {
  Serial.begin(115200);
  Wire.begin();
  delay(1000);  // wait for the BNO055 to boot (needs about 650 ms)
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

// Nothing to do after setup().
void loop() {}

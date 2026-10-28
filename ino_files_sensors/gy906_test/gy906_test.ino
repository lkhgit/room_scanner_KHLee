// GY-906 (MLX90614) 적외선 온도 센서 테스트 - Arduino Uno
// 배선: VIN -> 5V (모듈에 레귤레이터가 없으면 3.3V), GND -> GND, SCL -> A5, SDA -> A4
// 필요 라이브러리: "Adafruit MLX90614 Library" (라이브러리 매니저에서 설치)

#include <Wire.h>
#include <Adafruit_MLX90614.h>

Adafruit_MLX90614 mlx = Adafruit_MLX90614();

// I2C 버스에 연결된 장치 주소를 찾아 출력 (MLX90614는 보통 0x5A)
void scanI2C() {
  Serial.println("I2C 장치 검색 중...");
  int found = 0;
  for (byte addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.print("  발견: 0x");
      if (addr < 16) Serial.print("0");
      Serial.println(addr, HEX);
      found++;
    }
  }
  if (found == 0) Serial.println("  장치 없음 -> 배선(SDA/SCL 바뀜, 전원)을 확인하세요");
}

void setup() {
  Serial.begin(9600);
  while (!Serial) {}
  Wire.begin();
  scanI2C();

  if (!mlx.begin()) {
    Serial.println("MLX90614 연결 실패. 배선과 주소(0x5A)를 확인하세요.");
    while (1) {}
  }
  Serial.print("방사율(emissivity): ");
  Serial.println(mlx.readEmissivity());
  Serial.println("주변온도(C)\t물체온도(C)");
}

void loop() {
  float ambient = mlx.readAmbientTempC();  // 센서 자체(주변) 온도
  float object  = mlx.readObjectTempC();   // 센서가 바라보는 물체 표면 온도

  Serial.print(ambient, 2);
  Serial.print("\t\t");
  Serial.println(object, 2);
  delay(500);
}

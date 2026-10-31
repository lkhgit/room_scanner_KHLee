// GY-906 (MLX90614) infrared temperature sensor test - Arduino Uno
// Wiring: VIN -> 5V (3.3V if the module has no regulator), GND -> GND, SCL -> A5, SDA -> A4
// Library needed: "Adafruit MLX90614 Library" (install it from the Library Manager)

#include <Wire.h>
#include <Adafruit_MLX90614.h>

Adafruit_MLX90614 mlx = Adafruit_MLX90614();

// Finds and prints the addresses of the devices on the I2C bus (the MLX90614 is usually 0x5A).
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

// Scans the bus, starts the sensor and prints its emissivity.
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

// Prints the ambient and object temperature twice a second.
void loop() {
  float ambient = mlx.readAmbientTempC();  // temperature of the sensor itself (ambient)
  float object  = mlx.readObjectTempC();   // surface temperature of the object the sensor faces

  Serial.print(ambient, 2);
  Serial.print("\t\t");
  Serial.println(object, 2);
  delay(500);
}

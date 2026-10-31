// ============================================================
//  Pan-tilt rig motor control + compass sensor (Arduino UNO + ULN2003 x2 + 28BYJ-48 x2 + GY-BNO055)
//  - lower stage (pan)  IN1-IN4 -> D8, D9, D10, D11
//  - upper stage (tilt) IN1-IN4 -> D4, D5, D6, D7
//  - GY-BNO055   VIN -> 5V, GND -> GND, SDA -> A4, SCL -> A5   (I2C address 0x29, tries 0x28 if not found)
//    no libraries to install (uses only Wire and EEPROM)
//    At boot it loads the magnetometer calibration that bno055_compass.ino (ino_files_sensors/compass_sensor_calibration)
//    saved in EEPROM and writes it to the sensor.
//      (EEPROM 0-3: check value 0xB0055C01, 4-25: the 22 bytes of calibration registers 0x55-0x6A)
//    This sketch never writes to EEPROM, so the calibration survives uploading other sketches.
//
//  Serial commands (115200 baud, ending with a newline '\n'):
//    P <n>   move the pan motor by n half steps (negative = backwards)
//    T <n>   move the tilt motor by n half steps
//    S <us>  set the delay between half steps (µs, 800-10000)
//    H <0|1> keep the coils powered after a move (0 = power off after a move, the default)
//    R       power off both motors
//    N       one compass reading -> "N mx my mz ax ay az eh cal"
//              mx-mz: magnetic field (1/16 µT), ax-az: acceleration (1/100 m/s², points up when at rest)
//              eh: the sensor's own heading (1/16°, from magnetic north)
//              cal: calibration status byte (sys<<6 | gyro<<4 | accel<<2 | mag, each 0-3)
//            "ERR imu" if there is no sensor
//    I       sensor status -> "I <address> <calibration loaded 0|1>"  ("ERR imu" if there is no sensor)
//    ?       ping -> "READY"
//  Sends "OK" when a move is done and "ERR" for an unknown command.
// ============================================================
#include <Wire.h>
#include <EEPROM.h>

const uint8_t PAN_PINS[4]  = {8, 9, 10, 11};
const uint8_t TILT_PINS[4] = {4, 5, 6, 7};

// half-step sequence (IN1, IN2, IN3, IN4)
const uint8_t SEQ[8][4] = {
  {1,0,0,0}, {1,1,0,0}, {0,1,0,0}, {0,1,1,0},
  {0,0,1,0}, {0,0,1,1}, {0,0,0,1}, {1,0,0,1}
};

int8_t panPhase = 0;
int8_t tiltPhase = 0;
unsigned int stepDelayUs = 1500;   // half-step interval at cruising speed
bool holdTorque = false;           // by default the coils go off after a move to reduce heat

const long RAMP_STEPS = 120;       // length of the acceleration/deceleration ramp (half steps)
const unsigned int RAMP_EXTRA_US = 18;  // extra delay per step within the ramp

String lineBuf;

// ---------------- BNO055 ----------------
const uint8_t BNO_CHIP_ID   = 0x00;   // must read 0xA0
const uint8_t BNO_PAGE_ID   = 0x07;
const uint8_t BNO_ACC_X     = 0x08;   // acceleration X,Y,Z (LSB first)
const uint8_t BNO_MAG_X     = 0x0E;   // magnetic field X,Y,Z
const uint8_t BNO_EUL_H     = 0x1A;   // Euler heading
const uint8_t BNO_OFFSETS   = 0x55;   // start of the calibration registers (22 bytes)
const uint8_t BNO_CALIB     = 0x35;
const uint8_t BNO_UNIT_SEL  = 0x3B;
const uint8_t BNO_OPR_MODE  = 0x3D;
const uint8_t BNO_PWR_MODE  = 0x3E;
const uint8_t BNO_SYS_TRIG  = 0x3F;
const uint8_t MODE_CONFIG   = 0x00;
const uint8_t MODE_COMPASS  = 0x09;   // accelerometer + magnetometer (the same mode as during calibration)

// same storage format as bno055_compass.ino
const uint32_t CALIB_MAGIC     = 0xB0055C01UL;
const int      CALIB_ADDR_MAGIC = 0;
const int      CALIB_ADDR_DATA  = 4;
const uint8_t  CALIB_LEN        = 22;

uint8_t bnoAddr = 0;                  // 0 = no sensor
bool calibLoaded = false;

// Writes one BNO055 register; returns false on an I2C error.
bool bnoWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(bnoAddr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

// Reads n bytes starting at a BNO055 register; returns false on an I2C error.
bool bnoRead(uint8_t reg, uint8_t *buf, uint8_t n) {
  Wire.beginTransmission(bnoAddr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(bnoAddr, n) != n) return false;
  for (uint8_t i = 0; i < n; i++) buf[i] = Wire.read();
  return true;
}

// Looks for a BNO055 at addr by reading its chip id, retrying for about a second after power-up.
bool bnoProbe(uint8_t addr) {
  bnoAddr = addr;
  uint8_t id = 0;
  for (uint8_t tries = 0; tries < 10; tries++) {     // right after power-up it takes a while to answer
    if (bnoRead(BNO_CHIP_ID, &id, 1) && id == 0xA0) return true;
    delay(100);
  }
  bnoAddr = 0;
  return false;
}

// Writes the calibration saved in EEPROM to the sensor (only possible in CONFIG mode).
bool loadCalib() {
  uint32_t magic;
  EEPROM.get(CALIB_ADDR_MAGIC, magic);
  if (magic != CALIB_MAGIC) return false;
  for (uint8_t i = 0; i < CALIB_LEN; i++) {
    if (!bnoWrite(BNO_OFFSETS + i, EEPROM.read(CALIB_ADDR_DATA + i))) return false;
  }
  return true;
}

// Finds the BNO055, loads the saved calibration and switches it to compass mode.
void bnoInit() {
  Wire.begin();
#if defined(WIRE_HAS_TIMEOUT)
  Wire.setWireTimeout(3000, true);                   // so a disconnected sensor does not hang the board
#endif
  if (!bnoProbe(0x29) && !bnoProbe(0x28)) return;
  bnoWrite(BNO_OPR_MODE, MODE_CONFIG); delay(25);
  bnoWrite(BNO_PAGE_ID, 0);
  bnoWrite(BNO_PWR_MODE, 0x00);       delay(10);     // normal
  bnoWrite(BNO_SYS_TRIG, 0x80);       delay(10);     // use the external 32.768 kHz crystal (same as during calibration)
  bnoWrite(BNO_UNIT_SEL, 0x80);       delay(10);     // m/s², degrees
  calibLoaded = loadCalib();          delay(10);
  bnoWrite(BNO_OPR_MODE, MODE_COMPASS); delay(30);
}

// Combines two little-endian bytes into a signed 16-bit value.
int16_t le16(const uint8_t *b) { return (int16_t)((uint16_t)b[0] | ((uint16_t)b[1] << 8)); }

// Sends one compass reading as "N mx my mz ax ay az eh cal", or "ERR imu".
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

// ---------------- motors ----------------
// Drives a motor's four coils for one step of the sequence.
void writeCoils(const uint8_t* pins, int8_t phase) {
  for (uint8_t i = 0; i < 4; i++) digitalWrite(pins[i], SEQ[phase][i]);
}

// Switches a motor's coils off.
void releaseCoils(const uint8_t* pins) {
  for (uint8_t i = 0; i < 4; i++) digitalWrite(pins[i], LOW);
}

// Moves a motor by the given number of half steps, slower at the start and end.
void moveMotor(const uint8_t* pins, int8_t &phase, long steps) {
  if (steps == 0) return;
  int8_t dir = (steps > 0) ? 1 : -1;
  long n = labs(steps);

  writeCoils(pins, phase);   // hold the current phase first so the first step does not jump
  delay(5);

  for (long i = 0; i < n; i++) {
    phase = (phase + dir + 8) % 8;
    writeCoils(pins, phase);

    // slow at the start and end (keeps the motor from skipping steps)
    long fromEdge = min(i, n - 1 - i);
    unsigned int d = stepDelayUs;
    if (fromEdge < RAMP_STEPS) d += (unsigned int)(RAMP_STEPS - fromEdge) * RAMP_EXTRA_US;
    delayMicroseconds(d);
  }

  delay(20);
  if (!holdTorque) releaseCoils(pins);
}

// Runs one serial command (P, T, S, H, R, N, I or ?).
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

// Sets up the motor pins, the serial port and the compass sensor, then reports READY.
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

// Collects serial characters into lines and handles each complete line.
void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') { handleLine(lineBuf); lineBuf = ""; }
    else if (c != '\r') lineBuf += c;
  }
}

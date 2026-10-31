/*
 * Surface temperature/humidity device - IR scan of the 20 faces of an icosahedron (data collection firmware v2)
 *
 * Hardware
 *   lower stage (pan)  28BYJ-48 + ULN2003 : IN1..IN4 -> D11, D10, D9, D8
 *   upper stage (tilt) 28BYJ-48 + ULN2003 : IN1..IN4 -> D7, D6, D5, D4  (assumes IN4 on D4)
 *   GY-906 (MLX90614, 0x5A), SHT41 (0x44), [optional] BNO055 (0x28 or 0x29)
 *   shared I2C: SDA->A4, SCL->A5
 *
 * Coordinates
 *   The pose at power-on (sensor facing forward, level) is azimuth 0°, elevation 0°.
 *   The icosahedron has one vertex pointing at the zenith.
 *   5 upper faces at +52.62°, 10 middle faces at ±10.81°, 5 lower faces at -52.62°, in 10 columns 36° apart.
 *
 * Serial protocol (115200 bps, one message per line)
 *   Lines starting with '#' are meta information, all others are data rows.
 *   #HDR,<column names>              data column definition (once at start)
 *   #SESSION,heading_deg,<value>,mag_calib,<0-3>,sys_calib,<0-3>
 *                                     device heading (with a BNO055, measured before the motors are powered)
 *   #SCAN_BEGIN,<scan_id>
 *   <scan_id>,,<face>,<zone>,<az>,<el>,<t_surf>,<t_surf_sd>,<t_air>,<rh>,<t_sensor>   x 20
 *   #SCAN_END,<scan_id>
 *   #ABORT,<scan_id>                  when a scan is stopped (no data rows)
 *   #INFO,<message>                   human-readable notes
 *   The timestamp column of the data rows is left empty; the PC logger fills it in.
 *
 * Serial commands
 *   At power-on the device waits without moving the motors (prints #INFO,ready).
 *   n : one scan (then returns home and waits again)
 *   s : stop the scan at once -> return home -> wait
 *   For repeated measurements the PC logger sends n at the chosen interval.
 *
 * Backlash: every target position is approached from the + direction only (tilt + is up).
 * Sag: the tilt coils stay powered during a scan, and tilt 0° is set again at the start of each scan.
 * Measurement: 2 s wait after each move, then 10 readings over about 1 s, averaged. One scan takes about 1.5 minutes.
 *
 * Libraries: Adafruit MLX90614, Adafruit SHT4x, [optional] Adafruit BNO055
 */

#include <Wire.h>
#include <math.h>
#include <Adafruit_MLX90614.h>
#include <Adafruit_SHT4x.h>

#define USE_BNO055 0                   // set to 1 if a BNO055 is fitted

#if USE_BNO055
#include <Adafruit_BNO055.h>
#endif

// ---------------- settings ----------------
const uint8_t PAN_PINS[4]  = {11, 10, 9, 8};
const uint8_t TILT_PINS[4] = {7, 6, 5, 4};

const float STEPS_PER_REV = 4096.0;    // 28BYJ-48 in half steps (may measure closer to 4076)
const int   PAN_DIR  = 1;              // -1 if it turns the wrong way
const int   TILT_DIR = 1;              // -1 if a + angle does not look up
const long  BACKLASH_OVERSHOOT = 80;   // comfortably larger than the backlash (steps)
const unsigned int  STEP_US   = 1200;  // time between steps (us)
// IR timing: the MLX90614's default filter settles in about 0.85 s (Melexis filter application note).
// Wait more than twice that, then average 10 readings over about 1 s.
const unsigned long SETTLE_MS = 2000;  // wait after a move (vibration dies down + filter settles)
const int   N_SAMPLES = 10;            // number of IR readings averaged
const unsigned long SAMPLE_INTERVAL_MS = 100; // time between readings
// Against tilt sag the tilt coils stay powered during a scan.
// true keeps them powered while waiting at home too (the motor keeps warming up and may heat the IR sensor).
const bool  HOLD_TILT_WHEN_IDLE = false;
const bool  COL0_UPPER = true;         // true if the azimuth 0° column holds the two upper faces

#if USE_BNO055
const uint8_t BNO_ADDR = 0x28;         // change if the I2C scanner reports 0x29
const unsigned long BNO_CALIB_TIMEOUT_MS = 60000UL; // longest wait for the magnetometer calibration
#endif

// ---------------- sensors ----------------
Adafruit_MLX90614 mlx;
Adafruit_SHT4x    sht4;
#if USE_BNO055
Adafruit_BNO055 bno(55, BNO_ADDR);
#endif

// ---------------- state ----------------
volatile bool stopReq = false;
bool aligned = false;                  // whether the backlash has been taken up
bool scanning = false;
bool scanNowReq = false;
unsigned long scanCount = 0;           // scan_id (starts at 1 at every power-on)

// Reads serial commands: s stops a running scan, n requests a scan when idle.
void pollSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if ((c == 's' || c == 'S') && scanning) stopReq = true;
    else if ((c == 'n' || c == 'N') && !scanning) scanNowReq = true;  // n is ignored during a scan
  }
}

// ---------------- stepper ----------------
const uint8_t SEQ[8][4] = {
  {1,0,0,0},{1,1,0,0},{0,1,0,0},{0,1,1,0},
  {0,0,1,0},{0,0,1,1},{0,0,0,1},{1,0,0,1}
};

// One 28BYJ-48 stepper on a ULN2003 driver; the position is counted in half steps.
struct Stepper28 {
  const uint8_t* pins;
  int    dir;
  long   pos;
  int8_t phase;

  // Sets the pins as outputs, zeroes the position and switches the coils off.
  void begin() {
    for (int k = 0; k < 4; k++) pinMode(pins[k], OUTPUT);
    pos = 0; phase = 0;
    release();
  }
  // Drives the coils for the current phase.
  void writePhase() {
    for (int k = 0; k < 4; k++) digitalWrite(pins[k], SEQ[phase][k]);
  }
  // Moves one half step in direction s (+1 or -1).
  void stepOnce(int s) {
    phase = (phase + s * dir + 8) % 8;
    writePhase();
    pos += s;
    delayMicroseconds(STEP_US);
  }
  // Steps straight to target, checking for serial commands on the way.
  void moveRaw(long target) {
    while (pos != target) {
      stepOnce(target > pos ? 1 : -1);
      pollSerial();
    }
  }
  void moveTo(long target) {           // Moves to target, always arriving from the + direction (removes backlash).
    if (target < pos) moveRaw(target - BACKLASH_OVERSHOOT);
    moveRaw(target);
  }
  // Moves to an angle in degrees.
  void moveToDeg(float deg) {
    moveTo(lround(deg / 360.0 * STEPS_PER_REV));
  }
  // Goes below 0 and back to 0 so the gears are engaged in the + direction.
  void takeUpBacklash() {
    writePhase();
    delay(5);
    moveRaw(-BACKLASH_OVERSHOOT);
    moveRaw(0);
  }
  // Switches all coils off.
  void release() {
    for (int k = 0; k < 4; k++) digitalWrite(pins[k], LOW);
  }
};

Stepper28 pan  = {PAN_PINS,  PAN_DIR,  0, 0};
Stepper28 tilt = {TILT_PINS, TILT_DIR, 0, 0};

// ---------------- face list ----------------
const int N_FACES = 20;
float faceAz[N_FACES], faceEl[N_FACES];
char  faceZone[N_FACES];
float faceT[N_FACES], faceSD[N_FACES], faceTs[N_FACES];

// Fills in the azimuth, elevation and zone of the 20 faces in scan order.
void buildFaces() {
  const float E_MID = atan((3.0 - sqrt(5.0)) / 4.0) * 180.0 / PI;  // 10.812°
  const float E_CAP = atan((3.0 + sqrt(5.0)) / 4.0) * 180.0 / PI;  // 52.623°
  int n = 0;
  for (int j = 0; j < 10; j++) {
    float az = 36.0 * j;
    bool upper = (j % 2 == 0) ? COL0_UPPER : !COL0_UPPER;
    if (upper) {                       // bottom -> top
      faceAz[n] = az; faceEl[n] =  E_MID; faceZone[n++] = 'M';
      faceAz[n] = az; faceEl[n] =  E_CAP; faceZone[n++] = 'T';
    } else {
      faceAz[n] = az; faceEl[n] = -E_CAP; faceZone[n++] = 'B';
      faceAz[n] = az; faceEl[n] = -E_MID; faceZone[n++] = 'M';
    }
  }
}

// ---------------- measurement ----------------
// Mean and standard deviation of the IR object temperature, and the mean temperature of the sensor itself.
void readIR(float &mean, float &sd, float &ts) {
  float s = 0, s2 = 0, sa = 0;
  for (int i = 0; i < N_SAMPLES; i++) {
    float v = mlx.readObjectTempC();
    s += v; s2 += v * v;
    sa += mlx.readAmbientTempC();
    delay(SAMPLE_INTERVAL_MS);
    pollSerial();
  }
  mean = s / N_SAMPLES;
  float var = s2 / N_SAMPLES - mean * mean;
  sd   = sqrt(var > 0 ? var : 0);
  ts   = sa / N_SAMPLES;
}

// Returns to azimuth 0°, elevation 0° and switches off the coils that do not need to stay powered.
void goHome() {
  tilt.moveToDeg(0);                   // tilt moves with its coils kept powered
  pan.moveToDeg(0);                    // turns back the way it came so the cable does not twist
  pan.release();
  if (!HOLD_TILT_WHEN_IDLE) tilt.release();
}

// Prints the data column definition.
void printHeader() {
  Serial.println(F("#HDR,scan_id,timestamp,face,zone,az_deg,el_deg,t_surf_c,t_surf_sd_c,t_air_c,rh_pct,t_sensor_c"));
}

// Measures all 20 faces, returns home and prints the scan; prints #ABORT if stopped.
void runScan() {
  if (!aligned) {                      // take up the pan gear play once, before the first scan
    pan.takeUpBacklash();
    pan.release();
    aligned = true;
  }
  // The tilt may have sagged while its coils were off, so every scan raises it to 0° again from below.
  // From then on the tilt coils stay powered until the scan ends.
  tilt.takeUpBacklash();
  scanning = true;
  stopReq = false;
  scanCount++;
  Serial.print(F("#SCAN_BEGIN,")); Serial.println(scanCount);

  for (int i = 0; i < N_FACES; i++) {
    pan.moveToDeg(faceAz[i]);
    if (stopReq) break;
    tilt.moveToDeg(faceEl[i]);
    if (stopReq) break;
    pan.release();                     // pan (vertical axis) carries no gravity load, so its coils go off; tilt stays powered
    delay(SETTLE_MS);
    readIR(faceT[i], faceSD[i], faceTs[i]);
    if (stopReq) break;
  }

  if (stopReq) {
    stopReq = false;
    Serial.print(F("#ABORT,")); Serial.println(scanCount);
    goHome();
    scanning = false;
    Serial.println(F("#INFO,stopped at home. send n to scan again"));
    return;
  }

  goHome();

  sensors_event_t hum, temp;
  sht4.getEvent(&hum, &temp);
  float Ta = temp.temperature;
  float RH = hum.relative_humidity;

  for (int i = 0; i < N_FACES; i++) {
    Serial.print(scanCount);          Serial.print(F(",,"));
    Serial.print(i + 1);              Serial.print(',');
    Serial.print(faceZone[i]);        Serial.print(',');
    Serial.print(faceAz[i], 1);       Serial.print(',');
    Serial.print(faceEl[i], 2);       Serial.print(',');
    Serial.print(faceT[i], 2);        Serial.print(',');
    Serial.print(faceSD[i], 3);       Serial.print(',');
    Serial.print(Ta, 2);              Serial.print(',');
    Serial.print(RH, 1);              Serial.print(',');
    Serial.println(faceTs[i], 2);
  }
  Serial.print(F("#SCAN_END,")); Serial.println(scanCount);
  scanning = false;
}

#if USE_BNO055
// Measures the heading before the motors are powered: waits for magnetometer calibration 3 (up to the timeout), then takes the circular mean of 20 readings.
void measureHeading() {
  if (!bno.begin()) {
    Serial.println(F("#INFO,BNO055 not found. check address"));
    Serial.println(F("#SESSION,heading_deg,NaN,mag_calib,0,sys_calib,0"));
    return;
  }
  bno.setExtCrystalUse(true);
  Serial.println(F("#INFO,BNO055 calibrating. rotate the whole device slowly, then place it at home pose"));
  uint8_t sys = 0, gyr = 0, acc = 0, mag = 0;
  unsigned long t0 = millis();
  while (millis() - t0 < BNO_CALIB_TIMEOUT_MS) {
    bno.getCalibration(&sys, &gyr, &acc, &mag);
    if (mag == 3) break;
    delay(200);
  }
  delay(2000);                         // let it settle after being put down
  float sx = 0, cy = 0;
  for (int i = 0; i < 20; i++) {
    sensors_event_t ev;
    bno.getEvent(&ev);
    float h = ev.orientation.x * PI / 180.0;
    sx += sin(h); cy += cos(h);
    delay(50);
  }
  float heading = atan2(sx, cy) * 180.0 / PI;
  if (heading < 0) heading += 360.0;
  bno.getCalibration(&sys, &gyr, &acc, &mag);
  Serial.print(F("#SESSION,heading_deg,")); Serial.print(heading, 1);
  Serial.print(F(",mag_calib,")); Serial.print(mag);
  Serial.print(F(",sys_calib,")); Serial.println(sys);
}
#endif

// Starts the sensors, builds the face list, measures the heading if enabled, and reports ready.
void setup() {
  Serial.begin(115200);
  Wire.begin();

  pan.begin();                         // pins only, coils stay off
  tilt.begin();

  if (!mlx.begin()) { Serial.println(F("#INFO,ERROR MLX90614 not found")); while (1); }
  if (!sht4.begin()) { Serial.println(F("#INFO,ERROR SHT41 not found"));   while (1); }
  sht4.setPrecision(SHT4X_HIGH_PRECISION);
  sht4.setHeater(SHT4X_NO_HEATER);

  buildFaces();
  printHeader();

#if USE_BNO055
  measureHeading();                    // measured before the motors are powered
#endif

  // the motors do not move; wait for n
  Serial.println(F("#INFO,ready"));
}

// Checks for commands and runs a scan when one is requested.
void loop() {
  pollSerial();
  if (scanNowReq) {
    scanNowReq = false;
    runScan();
  }
}

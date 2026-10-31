// ============================================================
//  XIAO ESP32S3 Sense - JPEG camera over USB serial
//
//  Arduino IDE settings:
//    Board: XIAO_ESP32S3
//    PSRAM: "OPI PSRAM"
//    USB CDC On Boot: "Enabled"
//
//  Serial commands (ending with a newline '\n'):
//    C          take one photo -> "IMG <byte count>\n" + JPEG binary
//    F <size>   change the resolution (QVGA, VGA, SVGA, XGA, HD, SXGA, UXGA) -> "OK"
//    ?          ping -> "READY"
// ============================================================
#include "esp_camera.h"

// XIAO ESP32S3 Sense camera pins
#define PWDN_GPIO_NUM   -1
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM   10
#define SIOD_GPIO_NUM   40
#define SIOC_GPIO_NUM   39
#define Y9_GPIO_NUM     48
#define Y8_GPIO_NUM     11
#define Y7_GPIO_NUM     12
#define Y6_GPIO_NUM     14
#define Y5_GPIO_NUM     16
#define Y4_GPIO_NUM     18
#define Y3_GPIO_NUM     17
#define Y2_GPIO_NUM     15
#define VSYNC_GPIO_NUM  38
#define HREF_GPIO_NUM   47
#define PCLK_GPIO_NUM   13

bool camOk = false;
String lineBuf;

// Turns a resolution name into the camera's frame size, or FRAMESIZE_INVALID.
framesize_t parseSize(const String &s) {
  if (s == "QVGA") return FRAMESIZE_QVGA;
  if (s == "VGA")  return FRAMESIZE_VGA;
  if (s == "SVGA") return FRAMESIZE_SVGA;
  if (s == "XGA")  return FRAMESIZE_XGA;
  if (s == "HD")   return FRAMESIZE_HD;
  if (s == "SXGA") return FRAMESIZE_SXGA;
  if (s == "UXGA") return FRAMESIZE_UXGA;
  return FRAMESIZE_INVALID;
}

// Grabs and throws away n frames.
void discardFrames(int n) {
  for (int i = 0; i < n; i++) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
  }
}

// Sets up the camera (JPEG, buffers in PSRAM) and starts at the preview resolution.
bool initCamera() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer   = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM;  c.pin_d1 = Y3_GPIO_NUM;
  c.pin_d2 = Y4_GPIO_NUM;  c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;  c.pin_d5 = Y7_GPIO_NUM;
  c.pin_d6 = Y8_GPIO_NUM;  c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM;
  c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM;
  c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM;
  c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM;
  c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = FRAMESIZE_UXGA;      // initialise at the largest resolution so the buffers are big enough
  c.jpeg_quality = 10;                // lower = better quality (10-12 recommended)
  c.fb_count = 2;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;   // always the latest frame

  if (esp_camera_init(&c) != ESP_OK) return false;

  sensor_t *s = esp_camera_sensor_get();
  if (s->id.PID == OV3660_PID) {      // recent units sometimes ship with an OV3660
    s->set_vflip(s, 1);
    s->set_brightness(s, 1);
    s->set_saturation(s, -2);
  }
  s->set_framesize(s, FRAMESIZE_VGA); // start at the preview resolution
  discardFrames(3);
  return true;
}

// Takes a fresh photo and sends it as "IMG <bytes>" followed by the JPEG data.
void sendFrame() {
  // throw away any old frame still in the buffer and take a new one
  discardFrames(1);
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) { Serial.println("ERR capture"); return; }

  Serial.printf("IMG %u\n", (unsigned)fb->len);
  size_t sent = 0;
  while (sent < fb->len) {
    size_t chunk = min((size_t)4096, fb->len - sent);
    size_t w = Serial.write(fb->buf + sent, chunk);
    if (w == 0) { delay(1); continue; }
    sent += w;
  }
  Serial.flush();
  esp_camera_fb_return(fb);
}

// Runs one serial command: ?, C or F <size>.
void handleLine(String line) {
  line.trim();
  if (line.length() == 0) return;

  if (line == "?") { Serial.println(camOk ? "READY" : "ERR camera init"); return; }
  if (!camOk)      { Serial.println("ERR camera init"); return; }

  if (line == "C") { sendFrame(); return; }

  if (line.startsWith("F")) {
    String name = line.substring(1); name.trim();
    framesize_t fs = parseSize(name);
    sensor_t *s = esp_camera_sensor_get();
    if (fs == FRAMESIZE_INVALID || s->set_framesize(s, fs) != 0) { Serial.println("ERR size"); return; }
    discardFrames(3);  // drop the frames right after a resolution change (exposure settles)
    Serial.println("OK");
    return;
  }
  Serial.println("ERR");
}

// Opens the serial port, starts the camera and reports READY or the error.
void setup() {
  Serial.begin(921600);
  Serial.setTimeout(50);
  camOk = initCamera();
  Serial.println(camOk ? "READY" : "ERR camera init");
}

// Collects serial characters into lines and handles each complete line.
void loop() {
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n') { handleLine(lineBuf); lineBuf = ""; }
    else if (ch != '\r') lineBuf += ch;
  }
}

# room_scanner_KHLee

## Quick start

**Step 1. Download the repository.** Opening `index.html` on the GitHub website shows only its source code, so you need a copy on your computer:

    git clone https://github.com/lkhgit/room_scanner_KHLee.git

Or, on the GitHub page, click **Code → Download ZIP** and extract the ZIP file.

To see only the demo, you can download just the `docs` folder instead of the whole repository:

    git clone --depth 1 --filter=blob:none --sparse https://github.com/lkhgit/room_scanner_KHLee.git
    cd room_scanner_KHLee
    git sparse-checkout set docs

**Step 2. See the demo (no installation).** In the downloaded folder, open `docs` and double-click `index.html`. It opens in your browser (use Chrome or Edge). The page needs an internet connection.

**Step 3. Run the full app.** You need Python 3.10 or later. In a terminal at the repository root, run:

    py -m pip install -r requirements.txt
    cd app/room_scanner
    py server.py

Then open http://localhost:8000 in your browser.

**Step 4. Try it without hardware.** Choose **Upload Scan** on the home screen, then drag in [app/room_scanner/bundle.zip](app/room_scanner/bundle.zip).

**Step 5. Scan a real room.** First upload the `.ino` firmware to each board. See [Scanning a room](#scanning-a-room).

## 1. Setup

**Python packages** (Python 3.10+, run once from the repository root):

    py -m pip install -r requirements.txt

- [requirements.txt](requirements.txt) covers every part of the project. For the web app alone, [app/room_scanner/requirements.txt](app/room_scanner/requirements.txt) is enough.
- For GPU support, install `torch` with the command from https://pytorch.org first.
- On Windows, use `py` instead of `python` if `python` reports `No module named pip`.

**HorizonNet** (needed for the 3D room model in the Sunlight tab). It is not included in this repository:

    cd app/sunlight_scanner
    git clone https://github.com/sunset1995/HorizonNet.git

Download `resnet50_rnn__mp3d.pth` (linked in the HorizonNet README) into `app/sunlight_scanner/HorizonNet/ckpt/`.

**API keys** (optional). Enter them in [app/mold_scanner/kma.py](app/mold_scanner/kma.py). Do not commit real keys.

| Variable | Source | Used for |
|---|---|---|
| `KMA_SERVICE_KEY` | data.go.kr (Korea Meteorological Administration) | Weather data for the mold simulation |
| `KAKAO_REST_KEY` | Kakao Developers REST API key | Place search in the Flood tab |

## 2. Demo (docs)

Double-click [docs/index.html](docs/index.html). It shows a precomputed scan of one real classroom and runs without a server. An internet connection is needed for the 3D library and map tiles.

Uploading or processing scans, connecting devices, recalculating, and live weather or place search work only in the full app. See [docs/README.md](docs/README.md) for details.

## 3. Running the app

    cd app/room_scanner
    py server.py

Open http://localhost:8000. Press Ctrl+C to stop. Settings are in [app/room_scanner/config.yaml](app/room_scanner/config.yaml).

### Uploading a scan bundle

Home → Upload Scan, then drop a `.zip` bundle. An example is at [app/room_scanner/bundle.zip](app/room_scanner/bundle.zip). The format is defined in [app/room_scanner/rs/bundle.py](app/room_scanner/rs/bundle.py).

### Scanning a room

Upload the firmware to each board with the Arduino IDE first:

| Module | Board | Firmware |
|---|---|---|
| Pan-tilt rig + compass (GY-BNO055) | Arduino UNO | [rig_motor.ino](app/sunlight_scanner/rig_motor/rig_motor.ino) |
| Camera | XIAO ESP32S3 Sense | [xiao_cam_serial.ino](app/sunlight_scanner/xiao_cam_serial/xiao_cam_serial.ino) |
| Surface sensor | Arduino UNO | [surface_scan.ino](app/mold_scanner/surface_scan/surface_scan.ino) |

- **Compass calibration**: before uploading `rig_motor.ino`, upload [bno055_compass.ino](ino_files_sensors/compass_sensor_calibration/bno055_compass.ino) to the rig UNO. Move the sensor in a figure eight until the calibration is saved to EEPROM. The saved values remain when you upload `rig_motor.ino`.
- **Libraries**:
  - `bno055_compass.ino` needs Adafruit BNO055 and Adafruit Unified Sensor.
  - `surface_scan.ino` needs Adafruit MLX90614, Adafruit SHT4x, and Adafruit BNO055.
  - `rig_motor.ino` needs none.
- **Camera board**: select `XIAO_ESP32S3` in the ESP32 board package.

Then go to Home → Start Room Scan:

1. **Room info**: enter a name. The location, latitude and longitude, north heading, and camera height are optional.
2. **Photos**: connect the rig and camera ports. Level the camera with W/A/S/D, then press Enter to start the automatic capture (about 21 shots).
3. **Surface scan**: swap in the surface module. Put it in the same spot, facing the camera's starting direction. Choose the port and wall material, then press Start (20 faces, about 1.5 minutes).
4. **Create room**: open the results (Sunlight, Mold, and Flood).

Check COM ports in Device Manager or the Arduino IDE. Close any other program using the port, such as the Serial Monitor. Raw data and a bundle you can upload again are saved in `app/room_scanner/scans/<id>/`. For the full procedure, see [app/room_scanner/README.md](app/room_scanner/README.md).

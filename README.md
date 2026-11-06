# Room Scanner – Semi-basement Room Evaluator

A pan-tilt scanning rig and web app that evaluates a semi-basement room for three risks that are hard to judge on a short viewing: **lack of sunlight, mold growth, and flooding**.

**[▶ Demo video (2 min)](https://youtu.be/WD72YnWC070)** · **[🌐 Live demo](https://lkhgit.github.io/room_scanner_KHLee/)** (precomputed scan, runs in the browser)

[![Demo video](https://img.youtube.com/vi/WD72YnWC070/hqdefault.jpg)](https://youtu.be/WD72YnWC070)

## Why

In the 1970s, amid military tension with North Korea, many residential buildings in Seoul were required to include basements that could serve as shelters. Over time, these spaces were converted into low-cost housing, and today semi-basement homes (*banjiha*) are among the cheapest places to live in the city.

Their structure makes poor living conditions common: little sunlight, damp walls that grow mold, and a high risk of flooding during heavy rain. These problems are hard to notice during a short visit before signing a lease. Room Scanner was built to measure and evaluate them, so that a prospective tenant can see the risks before moving in.

## How it works

### Sun Calculator

1. **Scan the room.** A pan-tilt rig takes about 21 photos in three tilt rows. They are projected into an equirectangular panorama using the motor angles, while a BNO055 compass records north.
2. **Rebuild the room in 3D.** HorizonNet predicts the wall boundaries and corners. False corners are filtered out, and the camera height scales the layout into a metric 3D model.
3. **Find the windows.** OneFormer (ADE20K) segments a face-on texture of each wall. Each "windowpane" blob's bounding box is cut out of the wall mesh as an opening.
4. **Locate the sun.** Solar altitude and azimuth come from Spencer's (1971) Fourier series, rotated by the room's north heading into model coordinates.
5. **Trace the sunlight.** Rays are cast from every mesh vertex toward the sun (Möller–Trumbore). Unblocked vertices are sunlit, with irradiance = clear-sky DNI × cos(incidence angle).

### Mold Simulator

1. **Measure the surfaces.** An MLX90614 infrared thermometer reads surface temperature in the 20 face directions of an imaginary icosahedron, and an SHT41 sensor records air temperature and humidity.
2. **Map measurements onto the room.** Each point on the 3D model takes the value of the nearest face direction of the north-aligned icosahedron, blended at the seams.
3. **Simulate surface temperatures over a year.** Past KMA weather stands in for future weather. With a temperature factor `f` fitted from the scan and ground temperature `θg`, a surface follows `θg + f(θi − θg)` during the user's heating or cooling, and `θg` otherwise.
4. **Estimate surface humidity.** Indoor vapor pressure is the outdoor value plus a scan-fitted moisture excess that grows as outdoor temperature falls. Surface relative humidity uses the saturation pressure at each surface's temperature.
5. **Predict mold growth.** The VTT model advances a mold index (0–6) hourly from each surface's temperature, humidity, and material, for a central and a worst-case scenario.

### Flood Risk Estimator

1. **Collect the data.** Fourteen years of Seoul's flood trace maps were gridded into 100 m cells and combined with hourly rainfall from 27 Korea Meteorological Administration stations.
2. **Simplify the rainfall.** Hierarchical clustering on rainfall correlation grouped the stations into five regions. Each year's regional 1-hour and 24-hour maxima were reduced by PCA to three principal components to avoid overfitting.
3. **Group the neighborhoods.** HDBSCAN grouped the 13,054 cells with flood history into 11 clusters by the rainfall of the years they flooded. Cells that never flooded have no record.
4. **Train the prediction model.** A Bayesian hierarchical logistic regression (PyMC) learns how likely each cluster is to flood, given a year's rainfall components. Leave-one-year-out cross-validation chose the setup.
5. **Calculate local risk.** Each cell's flood odds equal its cluster's odds multiplied by a cell-specific factor from its flood history, evaluated over 500 posterior samples. The map updates as you adjust the rainfall sliders.

## Quick start

**Step 1. See the demo online (no download, no installation).** Open https://lkhgit.github.io/room_scanner_KHLee/ in your browser (use Chrome or Edge). The page needs an internet connection.

**Step 2. Download the repository** (only needed to run the full app). Opening files on the GitHub website shows only their source code, so you need a copy on your computer:

    git clone --recurse-submodules https://github.com/lkhgit/room_scanner_KHLee.git

Or, on the GitHub page, click **Code → Download ZIP** and extract the ZIP file. The ZIP does not include HorizonNet (a submodule), so also download https://github.com/lkhgit/HorizonNet (**Code → Download ZIP**) and extract its contents into `app/sunlight_scanner/HorizonNet/`.

**Step 3. Run the full app.** You need Python 3.10 or later. In a terminal at the repository root, run:

    py -m pip install -r requirements.txt
    cd app/room_scanner
    py server.py

Then open http://localhost:8000 in your browser.

Before you upload or scan, enter your API keys in [app/mold_scanner/kma.py](app/mold_scanner/kma.py). Without them, the mold simulation and place search fail. See [API keys](#api-keys).

**Step 4. Try it without hardware.** Choose **Upload Scan** on the home screen, then drag in [app/room_scanner/bundle.zip](app/room_scanner/bundle.zip).

**Step 5. Scan a real room.** First upload the `.ino` firmware to each board. See [Scanning a room](#scanning-a-room).

## 1. Setup

**Python packages** (Python 3.10+, run once from the repository root):

    py -m pip install -r requirements.txt

- [requirements.txt](requirements.txt) covers every part of the project. For the web app alone, [app/room_scanner/requirements.txt](app/room_scanner/requirements.txt) is enough.
- For GPU support, install `torch` with the command from https://pytorch.org first.
- On Windows, use `py` instead of `python` if `python` reports `No module named pip`.

**HorizonNet** (needed for the 3D room model in the Sunlight tab) is a git submodule at `app/sunlight_scanner/HorizonNet/`. Clone with submodules:

    git clone --recurse-submodules https://github.com/lkhgit/room_scanner_KHLee.git

If you already cloned without it, run this in the repository root:

    git submodule update --init

If you downloaded the ZIP instead, download https://github.com/lkhgit/HorizonNet as a ZIP too and extract its contents into `app/sunlight_scanner/HorizonNet/` (so that `model.py` is directly inside that folder).

The submodule is [a fork](https://github.com/lkhgit/HorizonNet) of [sunset1995/HorizonNet](https://github.com/sunset1995/HorizonNet) (MIT License) with small patches for Python 3.12+ / torch 2.6+. The weights `resnet50_rnn__mp3d.pth` are downloaded from Hugging Face on the first run (about 330 MB) into `app/sunlight_scanner/HorizonNet/ckpt/`.

### API keys

The keys in this repository are left empty on purpose. Enter your own keys near the top of [app/mold_scanner/kma.py](app/mold_scanner/kma.py):

    KMA_SERVICE_KEY = "your data.go.kr service key"
    KAKAO_REST_KEY = "your Kakao REST API key"

| Variable | Where to get it | Used for | Without it |
|---|---|---|---|
| `KMA_SERVICE_KEY` | [data.go.kr](https://www.data.go.kr): sign up, then apply for the KMA APIs "ASOS hourly data" (종관기상관측 시간자료), "ASOS daily data" (종관기상관측 일자료), and "short-term forecast" (단기예보 조회서비스). One service key works for all three. | Weather data for the mold simulation; outdoor air at scan time | The Mold tab fails with "The weather-service key is not set". |
| `KAKAO_REST_KEY` | [Kakao Developers](https://developers.kakao.com): create an app and copy its **REST API key** (not the JavaScript key). | Place search in the Flood tab; finding coordinates from a place name | Place search does not work. You can still click the map to pick a spot. |

Restart the server after entering the keys. Do not commit real keys to git.

### Running without API keys

The app starts and runs without any keys. Only the features that call the weather or place-search service are affected:

| Feature | Without keys |
|---|---|
| Sunlight tab | Works. |
| Flood tab | Works. Place search is unavailable (no `KAKAO_REST_KEY`), so click the map to pick a spot. |
| Mold tab | Does not work. The simulation needs weather data and fails with "The weather-service key is not set". |
| Uploading a bundle, scanning, 3D room model | Work. |

The online demo needs no keys.

## 2. Demo (docs)

Open the live demo: https://lkhgit.github.io/room_scanner_KHLee/ (served by GitHub Pages from the [docs](docs) folder). It shows a precomputed scan of one real classroom and runs without a server. An internet connection is needed for the 3D library and map tiles.

To view it offline, download the repository and double-click [docs/index.html](docs/index.html).

Uploading or processing scans, connecting devices, recalculating, and live weather or place search work only in the full app. See [docs/README.md](docs/README.md) for details.

## 3. Running the app

    cd app/room_scanner
    py server.py

Open http://localhost:8000. Press Ctrl+C to stop. Settings are in [app/room_scanner/config.yaml](app/room_scanner/config.yaml).

### 3D model build time

Building the 3D room model from the panorama can take a long time. While it runs, the tabs behave as follows:

- **Sunlight calculator**: shows a loading window until the model is ready.
- **Mold**: uses an icosahedron in place of the room model.
- **Flood risk**: works normally, because it does not depend on the room model.

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

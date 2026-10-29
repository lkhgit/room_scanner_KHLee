# Room Scanner — static demo

This folder is a static snapshot of the Room Scanner results screen for one real scanned room
(a classroom at Seoul Science High School). It runs in any browser with no Python server, scanning hardware or API keys.

**Tabs**
- **Sunlight**: direct sun through the detected windows, every 10 minutes, on the two solstices and two equinoxes.
- **Mold**: a year of hourly VTT mold index for 20 surface directions, shown on the 3D room model. It uses a cold-wall
  demo scenario (`_build/mold_scenario.yaml`): some measured surface temperatures and materials were adjusted.
- **Flood risk**: the Seoul flood-probability map at the scan location. Change the rainfall pattern and recalculate in the browser.
- **How it works**: step-by-step explanations of the Sun Calculator, the Mold Simulator and the Flood Risk Estimator.

**What only the full app does**: driving the scanning modules over USB, uploading and processing new scans, recalculating
after changing the location or heading, sunlight for any date, and live KMA weather and Kakao place search
(here, place search only knows "서울과학고등학교").

## How to open it
**Double-click `index.html` in this folder** (Chrome or Edge). Nothing needs to be installed or run.
An internet connection is needed for the 3D library and the map tiles.
(If you downloaded the repository as a ZIP, extract it first, then open `docs/index.html`.)

## How the snapshot is made
`index.html` and `shim.js` are written for the demo. Everything else is generated from a processed room by:

    py docs/_build/build_snapshot.py 27131762a022

The script imports the server's own code (`app/room_scanner/rs`) and only reads from `app/room_scanner/`. Browsers block
module scripts and `fetch()` on a page opened by double-click, so the script
- bundles the app code (`_build/demo-app.js` + the app's `mold.js`, and the flood map's modules) into classic
  scripts (`app.bundle.js`, `flood/flood.bundle.js`), and
- writes every data file (room, mold result, 3D model, flood data) as a script under `data/` and `flood/data/`.

`shim.js` then answers the app's `/api/...` requests from those scripts, so the app code itself is unchanged.

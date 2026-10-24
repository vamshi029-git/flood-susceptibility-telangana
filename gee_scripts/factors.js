/**************************************************************************
 * MODULE 3 — FLOOD CONDITIONING FACTORS + TRAINING DATASET (GEE)
 * Project: Multi-event flood susceptibility mapping (Telangana / AP)
 *
 * For every flood / non-flood point of every event, this script extracts
 * 14 conditioning factors (+ optional distance to roads) and exports ONE
 * CSV that is the machine-learning dataset.
 *
 * BEFORE RUNNING: upload the 3 point shapefiles from Google Drive as
 * assets (Assets tab → NEW → Shape files) and check the asset paths below.
 **************************************************************************/

/* ============================ 1. SETTINGS ============================ */

var CFG = {
  crs: 'EPSG:32644',      // UTM 44N, same as flood mapping
  scale: 30,              // sampling resolution (m)
  rainDays: 5,            // event rainfall = total of the 5 days before the flood image
  antecDays: 30,          // antecedent rainfall = 30 days before that
  streamUpaKm2: 10,       // a pixel is a "stream" if > 10 km² drains through it
  driveFolder: 'flood_project',
  roadsAsset: null        // optional: 'projects/flood-mapping-511811/assets/roads'
};

// floodDate = date of the Sentinel-1 flood image used for that event
var EVENTS = [
  {
    name: 'godavari_2022',
    points: 'projects/flood-mapping-511811/assets/godavari_2022_flood_points',
    floodDate: '2022-07-14'
  },
  {
    name: 'munneru_2024',
    points: 'projects/flood-mapping-511811/assets/munneru_2024_flood_points',
    floodDate: '2024-09-01'
  },
  {
    name: 'budameru_2024',
    points: 'projects/flood-mapping-511811/assets/budameru_2024_flood_points',
    floodDate: '2024-09-01'
  }
];

// Column order in the final CSV
var COLUMNS = ['event', 'lon', 'lat',
  'elevation', 'slope', 'aspect', 'curvature', 'twi', 'spi', 'hand',
  'dist_river', 'drain_dens', 'lulc', 'ndvi', 'soil_tex',
  'rain_event', 'rain_antec', 'label'];
if (CFG.roadsAsset) { COLUMNS.splice(COLUMNS.length - 1, 0, 'dist_road'); }


/* ===================== 2. STATIC FACTORS (same for all) ============== */

// --- Terrain (Copernicus DEM 30 m) ---
var demCol = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM');
var dem = demCol.mosaic().setDefaultProjection(demCol.first().projection());

var elevation = dem.rename('elevation');
var slope = ee.Terrain.slope(dem).rename('slope');            // degrees
var aspect = ee.Terrain.aspect(dem).rename('aspect');         // 0–360°
// Laplacian: positive = concave (bowl, collects water), negative = convex
var curvature = dem.convolve(ee.Kernel.laplacian8({normalize: false}))
  .rename('curvature');

// --- Hydrology (MERIT Hydro, ~90 m) ---
var merit = ee.Image('MERIT/Hydro/v1_0_1');
var upa = merit.select('upa');                                // upstream area (km²)
var hand = merit.select('hnd').rename('hand');                // height above nearest drainage (m)

// TWI = ln(a / tanβ), SPI = ln(a · tanβ + 1); a = specific catchment area (m²/m)
var tanB = slope.multiply(Math.PI / 180).tan().max(0.001);
var sca = upa.multiply(1e6).divide(90);
var twi = sca.divide(tanB).log().rename('twi');
var spi = sca.multiply(tanB).add(1).log().rename('spi');

// Stream network from MERIT upstream area
var streams = upa.gt(CFG.streamUpaKm2).unmask(0)
  .reproject({crs: CFG.crs, scale: 90});

// Distance to nearest stream (m)
var distRiver = streams.fastDistanceTransform(256).sqrt().multiply(90)
  .unmask(256 * 90).rename('dist_river');

// Drainage density (km of stream per km², 1 km radius window)
var drainDens = streams.reduceNeighborhood({
  reducer: ee.Reducer.sum(),
  kernel: ee.Kernel.circle({radius: 1000, units: 'meters'})
}).multiply(0.09).divide(Math.PI).rename('drain_dens');

// --- Land cover (ESA WorldCover 2021, 10 m; categorical) ---
// 10 tree, 20 shrub, 30 grass, 40 crop, 50 built-up, 60 bare,
// 70 snow, 80 water, 90 wetland, 95 mangrove, 100 moss
var lulc = ee.ImageCollection('ESA/WorldCover/v200').first()
  .select('Map').rename('lulc');

// --- Soil texture (OpenLandMap USDA class at surface; categorical 1–12) ---
var soilTex = ee.Image('OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02')
  .select('b0').rename('soil_tex');

// --- Optional: distance to roads (uploaded OSM roads asset) ---
var distRoad = null;
if (CFG.roadsAsset) {
  var roads = ee.FeatureCollection(CFG.roadsAsset);
  distRoad = ee.Image(0).byte().paint(roads, 1)
    .reproject({crs: CFG.crs, scale: 30})
    .fastDistanceTransform(256).sqrt().multiply(30)
    .unmask(256 * 30).rename('dist_road');
}


/* =================== 3. EVENT-SPECIFIC FACTORS ======================= */

// Sentinel-2 cloud/shadow mask using the scene classification band
function maskS2(img) {
  var scl = img.select('SCL');
  var clear = scl.neq(3).and(scl.neq(8)).and(scl.neq(9))
    .and(scl.neq(10)).and(scl.neq(11));
  return img.updateMask(clear);
}

function eventFactors(ev, region) {
  var floodDay = ee.Date(ev.floodDate);

  // NDVI: median of the 12 months before the flood (cloud-masked)
  var ndvi = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(region)
    .filterDate(floodDay.advance(-1, 'year'), floodDay)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    .map(maskS2)
    .median()
    .normalizedDifference(['B8', 'B4'])
    .rename('ndvi');

  // Event rainfall: IMERG half-hourly (mm/h) → total mm over rainDays
  var rainStart = floodDay.advance(-CFG.rainDays, 'day');
  var rainEvent = ee.ImageCollection('NASA/GPM_L3/IMERG_V07')
    .filterDate(rainStart, floodDay)
    .select('precipitation')
    .sum().multiply(0.5)
    .rename('rain_event');

  // Antecedent rainfall: CHIRPS daily total for the antecDays before that
  var rainAntec = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
    .filterDate(rainStart.advance(-CFG.antecDays, 'day'), rainStart)
    .select('precipitation')
    .sum()
    .rename('rain_antec');

  return [ndvi, rainEvent, rainAntec];
}


/* ========================= 4. BUILD DATASET ========================== */

var allSamples = ee.FeatureCollection([]);

EVENTS.forEach(function (ev, i) {
  var pts = ee.FeatureCollection(ev.points);
  var region = pts.geometry().bounds();

  var bands = [elevation, slope, aspect, curvature, twi, spi, hand,
    distRiver, drainDens, lulc, soilTex].concat(eventFactors(ev, region));
  if (distRoad) { bands.push(distRoad); }
  var stack = ee.Image.cat(bands);

  var samples = stack.sampleRegions({
    collection: pts,
    properties: ['label', 'event', 'lon', 'lat'],
    scale: CFG.scale,
    projection: CFG.crs,
    tileScale: 8,
    geometries: false
  });

  print(ev.name + ' — input points:', pts.size(),
        ' samples with all factors:', samples.size());
  print(ev.name + ' — samples per class (0 / 1):', samples.aggregate_histogram('label'));

  allSamples = allSamples.merge(samples);

  // Visual check for the first event only
  if (i === 0) {
    Map.centerObject(region, 9);
    Map.addLayer(elevation.clip(region), {min: 0, max: 400,
      palette: ['006600', 'ffff00', '993300', 'ffffff']}, 'Elevation', false);
    Map.addLayer(hand.clip(region), {min: 0, max: 50,
      palette: ['0000ff', 'ffffff', 'ff0000']}, 'HAND', false);
    Map.addLayer(distRiver.clip(region), {min: 0, max: 5000,
      palette: ['08306b', 'ffffff']}, 'Distance to river', false);
    Map.addLayer(stack.select('rain_event').clip(region), {min: 0, max: 300,
      palette: ['ffffff', '6baed6', '08306b']}, 'Event rainfall (mm)', false);
    Map.addLayer(pts.filter(ee.Filter.eq('label', 1)), {color: 'red'},
      ev.name + ' flood points');
    Map.addLayer(pts.filter(ee.Filter.eq('label', 0)), {color: 'green'},
      ev.name + ' non-flood points');
  }
});

print('TOTAL samples:', allSamples.size());
print('Preview (first 5 rows):', allSamples.limit(5));
Map.setOptions('HYBRID');


/* ============================ 5. EXPORT ============================== */

Export.table.toDrive({
  collection: allSamples,
  description: 'flood_dataset_all_events',
  folder: CFG.driveFolder,
  fileFormat: 'CSV',
  selectors: COLUMNS
});

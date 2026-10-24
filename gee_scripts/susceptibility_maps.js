/**************************************************************************
 * MODULE 7 — FLOOD SUSCEPTIBILITY MAPS (GEE)
 * Trains a Random Forest (200 trees, same as Python) on all 3 flood events,
 * then maps flood probability for:
 *   - the 3 study areas (with their real event rainfall)  → validation maps
 *   - Hyderabad, Warangal, all of Telangana (rainfall scenario) → application maps
 **************************************************************************/

/* ============================ 1. SETTINGS ============================ */

var CFG = {
  crs: 'EPSG:32644',
  rainDays: 5,
  antecDays: 30,
  streamUpaKm2: 10,
  driveFolder: 'flood_project',
  // Rainfall scenario for areas WITHOUT a recorded flood event.
  // Defaults = median of the training data (a typical heavy monsoon event).
  scenario: {rain_event: 150, rain_antec: 265, ndviDate: '2024-09-01'}
};

// Which maps to run (indexes into MAPS below). Start small: [0] or [3].
var RUN = [3];

var TRAIN = [
  {name: 'godavari_2022', points: 'projects/flood-mapping-511811/assets/godavari_2022_flood_points', floodDate: '2022-07-14'},
  {name: 'munneru_2024',  points: 'projects/flood-mapping-511811/assets/munneru_2024_flood_points',  floodDate: '2024-09-01'},
  {name: 'budameru_2024', points: 'projects/flood-mapping-511811/assets/budameru_2024_flood_points', floodDate: '2024-09-01'}
];

// Telangana boundary from geoBoundaries (community catalog); falls back to a box
var tgFC = ee.FeatureCollection('projects/sat-io/open-datasets/geoboundaries/CGAZ_ADM1')
  .filter(ee.Filter.eq('shapeName', 'Telangana'));
var tgBox = ee.Geometry.Rectangle([77.2, 15.8, 81.4, 19.95]);
var telangana = ee.Geometry(ee.Algorithms.If(tgFC.size().gt(0), tgFC.geometry(), tgBox));

var MAPS = [
  // --- Validation maps: real rainfall of each flood event ---
  {name: 'godavari_2022', region: ee.Geometry.Rectangle([80.3, 17.2, 81.4, 18.3]),  floodDate: '2022-07-14', scale: 30},
  {name: 'munneru_2024',  region: ee.Geometry.Rectangle([79.8, 16.95, 80.9, 17.6]), floodDate: '2024-09-01', scale: 30},
  {name: 'budameru_2024', region: ee.Geometry.Rectangle([80.3, 16.3, 81.0, 16.9]),  floodDate: '2024-09-01', scale: 30},
  // --- Application maps: areas the model never trained on (rainfall scenario) ---
  {name: 'hyderabad', region: ee.Geometry.Rectangle([78.20, 17.20, 78.70, 17.60]), scale: 30},
  {name: 'warangal',  region: ee.Geometry.Rectangle([79.35, 17.80, 79.85, 18.15]), scale: 30},
  {name: 'telangana', region: telangana, scale: 90}
];

// Absolute elevation removed: it blocks transfer to the higher Deccan plateau
// (Hyderabad ~500 m) and RF without it scores the same on unseen floods
var BANDS = ['slope', 'aspect_sin', 'aspect_cos', 'curvature', 'twi', 'spi',
  'hand', 'dist_river', 'ndvi', 'rain_event', 'rain_antec', 'lulc', 'soil_tex'];


/* ===================== 2. STATIC FACTORS ============================= */

var demCol = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM');
var dem = demCol.mosaic().setDefaultProjection(demCol.first().projection());
var slope = ee.Terrain.slope(dem).rename('slope');
var aspectRad = ee.Terrain.aspect(dem).multiply(Math.PI / 180);
var curvature = dem.convolve(ee.Kernel.laplacian8({normalize: false})).rename('curvature');

var merit = ee.Image('MERIT/Hydro/v1_0_1');
var upa = merit.select('upa');
var tanB = slope.multiply(Math.PI / 180).tan().max(0.001);
var sca = upa.multiply(1e6).divide(90);

var streams = upa.gt(CFG.streamUpaKm2).unmask(0).reproject({crs: CFG.crs, scale: 90});

var STATIC = ee.Image.cat([
  dem.rename('elevation'),
  slope,
  aspectRad.sin().rename('aspect_sin'),
  aspectRad.cos().rename('aspect_cos'),
  curvature,
  sca.divide(tanB).log().rename('twi'),
  sca.multiply(tanB).add(1).log().rename('spi'),
  merit.select('hnd').rename('hand'),
  streams.fastDistanceTransform(256).sqrt().multiply(90).unmask(256 * 90).rename('dist_river'),
  ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').rename('lulc'),
  ee.Image('OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02').select('b0').rename('soil_tex')
]);


/* =================== 3. EVENT / SCENARIO FACTORS ===================== */

function maskS2(img) {
  var scl = img.select('SCL');
  return img.updateMask(scl.neq(3).and(scl.neq(8)).and(scl.neq(9))
    .and(scl.neq(10)).and(scl.neq(11)));
}

function ndviBefore(date, region) {
  var d = ee.Date(date);
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(region).filterDate(d.advance(-1, 'year'), d)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
    .map(maskS2).median()
    .normalizedDifference(['B8', 'B4']).rename('ndvi');
}

// Real rainfall before a flood (same definition as the training dataset)
function eventRain(date) {
  var d = ee.Date(date);
  var start = d.advance(-CFG.rainDays, 'day');
  var ev = ee.ImageCollection('NASA/GPM_L3/IMERG_V07').filterDate(start, d)
    .select('precipitation').sum().multiply(0.5).rename('rain_event');
  var an = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
    .filterDate(start.advance(-CFG.antecDays, 'day'), start)
    .select('precipitation').sum().rename('rain_antec');
  return ee.Image.cat([ev, an]);
}

// Full 14-band factor stack for a region
function buildStack(m) {
  if (m.floodDate) {
    return STATIC.addBands(ndviBefore(m.floodDate, m.region))
                 .addBands(eventRain(m.floodDate)).select(BANDS);
  }
  var rain = ee.Image.constant([CFG.scenario.rain_event, CFG.scenario.rain_antec])
    .rename(['rain_event', 'rain_antec']).toFloat();
  return STATIC.addBands(ndviBefore(CFG.scenario.ndviDate, m.region))
               .addBands(rain).select(BANDS);
}


/* ===================== 4. TRAIN RANDOM FOREST ======================== */

var training = ee.FeatureCollection([]);
TRAIN.forEach(function (t) {
  var pts = ee.FeatureCollection(t.points);
  var stack = buildStack({floodDate: t.floodDate, region: pts.geometry().bounds()});
  training = training.merge(stack.sampleRegions({
    collection: pts, properties: ['label'], scale: 30,
    projection: CFG.crs, tileScale: 8
  }));
});

var rf = ee.Classifier.smileRandomForest({
  numberOfTrees: 200, variablesPerSplit: 4, seed: 42
}).setOutputMode('PROBABILITY')
  .train({features: training, classProperty: 'label', inputProperties: BANDS});

print('Training samples:', training.size());
print('Random Forest out-of-bag error:', rf.explain().get('outOfBagErrorEstimate'));


/* ====================== 5. MAP + CLASSIFY ============================ */

var CLASS_NAMES = ['Very low', 'Low', 'Moderate', 'High', 'Very high'];
var PALETTE = ['1a9850', '91cf60', 'fee08b', 'fc8d59', 'd73027'];

function makeMap(m) {
  var prob = buildStack(m).classify(rf).rename('flood_prob').clip(m.region);

  // 5 classes: 0–0.2, 0.2–0.4, 0.4–0.6, 0.6–0.8, 0.8–1
  var cls = prob.multiply(5).floor().min(4).add(1).toByte().rename('class');

  Map.addLayer(prob, {min: 0, max: 1, palette: PALETTE}, m.name + ' | flood probability');
  Map.addLayer(cls, {min: 1, max: 5, palette: PALETTE}, m.name + ' | 5 classes', false);

  // % of area in each class
  var areas = ee.Image.pixelArea().divide(1e6).addBands(cls).reduceRegion({
    reducer: ee.Reducer.sum().group({groupField: 1, groupName: 'class'}),
    geometry: m.region, scale: m.scale * 3, crs: CFG.crs,
    maxPixels: 1e13, bestEffort: true, tileScale: 8
  });
  print(m.name + ' — area (km²) per class (1 = very low … 5 = very high):', areas.get('groups'));

  Export.image.toDrive({
    image: prob.toFloat(), description: m.name + '_flood_probability',
    folder: CFG.driveFolder, region: m.region, scale: m.scale,
    crs: CFG.crs, maxPixels: 1e13
  });
  Export.image.toDrive({
    image: cls.unmask(0), description: m.name + '_susceptibility_classes',
    folder: CFG.driveFolder, region: m.region, scale: m.scale,
    crs: CFG.crs, maxPixels: 1e13
  });
}

RUN.forEach(function (i) { makeMap(MAPS[i]); });
Map.centerObject(MAPS[RUN[0]].region, 10);
Map.setOptions('HYBRID');

// Legend
var legend = ui.Panel({style: {position: 'bottom-left', padding: '8px'}});
legend.add(ui.Label('Flood susceptibility', {fontWeight: 'bold'}));
CLASS_NAMES.forEach(function (name, i) {
  legend.add(ui.Panel([
    ui.Label('', {backgroundColor: '#' + PALETTE[i], padding: '8px', margin: '2px'}),
    ui.Label(name, {margin: '4px'})
  ], ui.Panel.Layout.Flow('horizontal')));
});
Map.add(legend);

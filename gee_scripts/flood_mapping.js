/**************************************************************************
 * FLOOD MAPPING + FLOOD INVENTORY (Sentinel-1 SAR) — Google Earth Engine
 * Project: Multi-event flood susceptibility mapping (Telangana / AP)
 **************************************************************************/

/* ============================ 1. SETTINGS ============================ */

var CFG = {
  crs: 'EPSG:32644',        // UTM zone 44N (covers Telangana / AP)
  scale: 30,                // output resolution in metres (same as base paper)
  speckleRadius: 50,        // focal median radius (m)
  diffThresholdDb: -3,      // post − pre must drop by at least 3 dB
  thrMin: -22, thrMax: -15, // accepted range for the Otsu water threshold (dB)
  thrDefault: -18,          // used if Otsu falls outside that range (common VV water value)
  maxSlope: 5,              // degrees; steeper pixels can't be flood
  maxHand: 20,              // metres above nearest drainage; higher can't be flood
  permWaterMonths: 10,      // JRC seasonality >= this = permanent water
  minPixels: 8,             // remove flood patches smaller than this (30 m pixels)
  nonFloodBuffer: 500,      // non-flood points must be >= 500 m from flood
  pointsPerClass: 1500,     // flood points = non-flood points = this number
  seed: 42,
  driveFolder: 'flood_project',
  showPoints: false         // true = draw points on the map (often times out; export still works)
};

// Which events to run: [0] = Godavari only, [0, 1, 2] = all three
var RUN = [0];

var EVENTS = [
  {
    name: 'godavari_2022',
    aoi: ee.Geometry.Rectangle([80.3, 17.2, 81.4, 18.3]),   // Bhadradri Kothagudem / Bhadrachalam
    pre:  ['2022-06-01', '2022-07-05'],
    post: ['2022-07-14', '2022-07-23']
  },
  {
    name: 'munneru_2024',
    aoi: ee.Geometry.Rectangle([79.8, 16.95, 80.9, 17.6]),   // Khammam
    pre:  ['2024-08-01', '2024-08-26'],
    post: ['2024-08-31', '2024-09-08']
  },
  {
    name: 'budameru_2024',
    aoi: ee.Geometry.Rectangle([80.3, 16.3, 81.0, 16.9]),    // Vijayawada (NTR district)
    pre:  ['2024-08-01', '2024-08-26'],
    post: ['2024-08-31', '2024-09-08']
  }
];

/* ========================= 2. STATIC DATASETS ======================== */

var S1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .select('VV');

// Permanent water (months per year with water, 0–12)
var permWater = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
  .select('seasonality').unmask(0)
  .gte(CFG.permWaterMonths);

// HAND: height above nearest drainage (m)
var hand = ee.Image('MERIT/Hydro/v1_0_1').select('hnd');

// Slope from Copernicus DEM (keep native projection so slope is correct)
var demCol = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM');
var dem = demCol.mosaic().setDefaultProjection(demCol.first().projection());
var slope = ee.Terrain.slope(dem);

/* ========================= 3. HELPER FUNCTIONS ======================= */

// Adds date + % of AOI covered to every image, for printing
function describe(col, aoi) {
  return col.map(function (img) {
    var cov = img.geometry().intersection(aoi, 1000).area(1000)
      .divide(aoi.area(1000)).multiply(100);
    return img.set({
      date: img.date().format('YYYY-MM-dd'),
      coverage_pct: cov
    });
  });
}

// Otsu threshold from a histogram (last bucket skipped to avoid divide-by-zero)
function otsu(histogram) {
  var counts = ee.Array(ee.Dictionary(histogram).get('histogram'));
  var means = ee.Array(ee.Dictionary(histogram).get('bucketMeans'));
  var size = means.length().get([0]);
  var total = counts.reduce(ee.Reducer.sum(), [0]).get([0]);
  var sum = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0]);
  var mean = sum.divide(total);
  var indices = ee.List.sequence(1, size.subtract(1));
  var bss = indices.map(function (i) {
    var aCounts = counts.slice(0, 0, i);
    var aCount = aCounts.reduce(ee.Reducer.sum(), [0]).get([0]);
    var aMeans = means.slice(0, 0, i);
    var aMean = aMeans.multiply(aCounts).reduce(ee.Reducer.sum(), [0]).get([0])
      .divide(aCount);
    var bCount = total.subtract(aCount);
    var bMean = sum.subtract(aCount.multiply(aMean)).divide(bCount);
    return aCount.multiply(aMean.subtract(mean).pow(2))
      .add(bCount.multiply(bMean.subtract(mean).pow(2)));
  });
  return means.slice(0, 0, size.subtract(1)).sort(bss).get([-1]);
}

/* ========================= 4. MAIN PROCESSING ======================== */

function processEvent(ev) {
  var aoi = ev.aoi;
  print('==================== ' + ev.name + ' ====================');

  // ---- 4.1 Image availability ----
  var postAll = describe(S1.filterBounds(aoi).filterDate(ev.post[0], ev.post[1]), aoi);
  var preAll = describe(S1.filterBounds(aoi).filterDate(ev.pre[0], ev.pre[1]), aoi);

  print('POST images (date / orbit / pass / % AOI):',
    postAll.aggregate_array('date'),
    postAll.aggregate_array('relativeOrbitNumber_start'),
    postAll.aggregate_array('orbitProperties_pass'),
    postAll.aggregate_array('coverage_pct'));
  print('PRE images (date / orbit / pass / % AOI):',
    preAll.aggregate_array('date'),
    preAll.aggregate_array('relativeOrbitNumber_start'),
    preAll.aggregate_array('orbitProperties_pass'),
    preAll.aggregate_array('coverage_pct'));

  // ---- 4.2 Pick the best-covering orbit that exists in BOTH pre and post ----
  var postMatched = postAll.map(function (img) {
    var nPre = preAll
      .filter(ee.Filter.eq('relativeOrbitNumber_start', img.get('relativeOrbitNumber_start')))
      .filter(ee.Filter.eq('orbitProperties_pass', img.get('orbitProperties_pass')))
      .size();
    return img.set('n_pre', nPre);
  }).filter(ee.Filter.gt('n_pre', 0));
  print('POST images with a matching PRE orbit (date / orbit):',
    postMatched.aggregate_array('date'),
    postMatched.aggregate_array('relativeOrbitNumber_start'));

  var best = ee.Image(postMatched.sort('coverage_pct', false).first());
  var orbit = best.get('relativeOrbitNumber_start');
  var pass = best.get('orbitProperties_pass');
  print('Selected orbit / pass:', orbit, pass);

  var sameOrbit = ee.Filter.and(
    ee.Filter.eq('relativeOrbitNumber_start', orbit),
    ee.Filter.eq('orbitProperties_pass', pass));
  var postCol = postAll.filter(sameOrbit);
  var preCol = preAll.filter(sameOrbit);
  print('Images used — pre:', preCol.size(), ' post:', postCol.size());

  var proj = best.projection();

  // Pre = median of all pre images (less speckle); post = earliest flood image on top
  var pre = preCol.median().setDefaultProjection(proj)
    .focal_median(CFG.speckleRadius, 'circle', 'meters').clip(aoi);
  var post = postCol.sort('system:time_start', false).mosaic().setDefaultProjection(proj)
    .focal_median(CFG.speckleRadius, 'circle', 'meters').clip(aoi);

  var valid = pre.mask().and(post.mask());

  // ---- 4.3 Water threshold (Otsu) ----
  // Histogram only within 3 km of permanent water, so it has both a water
  // and a land peak (Otsu needs a two-peaked histogram to work well)
  var nearWater = permWater.focal_max({radius: 3000, units: 'meters'});
  var hist = post.updateMask(nearWater).reduceRegion({
    reducer: ee.Reducer.histogram(255, 0.1),
    geometry: aoi, scale: 90, maxPixels: 1e10, bestEffort: true
  }).get('VV');
  var rawThr = ee.Number(otsu(hist));
  var otsuOk = rawThr.gte(CFG.thrMin).and(rawThr.lte(CFG.thrMax));
  var thr = ee.Number(ee.Algorithms.If(otsuOk, rawThr, CFG.thrDefault));
  print('Otsu threshold (raw, dB):', rawThr);
  print('Otsu in accepted range? (1 = yes, 0 = no → default used):', otsuOk);
  print('Water threshold USED (dB):', thr);

  // ---- 4.4 Flood = new water, outside permanent water, in flat low terrain ----
  var water = post.lt(thr);
  var changed = post.subtract(pre).lt(CFG.diffThresholdDb);

  var floodRaw = water.and(changed)
    .and(permWater.not())
    .and(slope.lt(CFG.maxSlope))
    .and(hand.lt(CFG.maxHand))
    .reproject({crs: CFG.crs, scale: CFG.scale});

  // Remove small speckle patches
  var connected = floodRaw.selfMask().connectedPixelCount(CFG.minPixels + 1, true);
  var flood = connected.gte(CFG.minPixels).unmask(0)
    .updateMask(valid).clip(aoi).rename('flood').toByte();

  // ---- 4.5 Flood area ----
  var areaKm2 = flood.multiply(ee.Image.pixelArea()).reduceRegion({
    reducer: ee.Reducer.sum(), geometry: aoi, scale: CFG.scale,
    crs: CFG.crs, maxPixels: 1e13, bestEffort: true
  }).get('flood');
  print('Flooded area (km²):', ee.Number(areaKm2).divide(1e6));

  // ---- 4.6 Labels for sampling: 1 = flood, 0 = safe non-flood ----
  var nearFlood = flood.focal_max({radius: CFG.nonFloodBuffer, units: 'meters'});
  var label = flood
    .where(flood.eq(0).and(nearFlood.eq(1)), 2)   // 2 = too close to flood → drop
    .updateMask(permWater.not());
  label = label.updateMask(label.neq(2)).rename('label').toInt();

  var points = label.stratifiedSample({
    numPoints: 0,
    classBand: 'label',
    classValues: [0, 1],
    classPoints: [CFG.pointsPerClass, CFG.pointsPerClass],
    region: aoi,
    scale: CFG.scale,
    projection: CFG.crs,
    seed: CFG.seed,
    geometries: true,
    tileScale: 4
  }).map(function (f) {
    var c = f.geometry().coordinates();
    return f.set({event: ev.name, lon: c.get(0), lat: c.get(1)});
  });
  print('Points per class (0 = non-flood, 1 = flood):', points.aggregate_histogram('label'));

  // ---- 4.7 Map layers ----
  Map.addLayer(pre, {min: -25, max: 0}, ev.name + ' | pre-flood VV (dB)', false);
  Map.addLayer(post, {min: -25, max: 0}, ev.name + ' | flood VV (dB)', false);
  Map.addLayer(permWater.selfMask().clip(aoi), {palette: ['000080']},
    ev.name + ' | permanent water', false);
  Map.addLayer(flood.selfMask(), {palette: ['00BFFF']}, ev.name + ' | FLOOD');
  if (CFG.showPoints) {
    Map.addLayer(points.filter(ee.Filter.eq('label', 1)), {color: 'red'},
      ev.name + ' | flood points', false);
    Map.addLayer(points.filter(ee.Filter.eq('label', 0)), {color: 'green'},
      ev.name + ' | non-flood points', false);
  }

  // ---- 4.8 Exports (Tasks tab → RUN) ----
  Export.image.toDrive({
    image: flood.unmask(255).toByte(),   // 1 = flood, 0 = not flooded, 255 = no data
    description: ev.name + '_flood_map',
    folder: CFG.driveFolder,
    region: aoi,
    scale: CFG.scale,
    crs: CFG.crs,
    maxPixels: 1e13
  });

  Export.table.toDrive({
    collection: points,
    description: ev.name + '_flood_points',
    folder: CFG.driveFolder,
    fileFormat: 'SHP'
  });
}

/* ============================== 5. RUN =============================== */

RUN.forEach(function (i) { processEvent(EVENTS[i]); });
Map.centerObject(EVENTS[RUN[0]].aoi, 9);
Map.setOptions('HYBRID');

"""Google Earth Engine (Python API) versions of the GEE scripts in gee_scripts/.

- flood_inventory(): Sentinel-1 flood mapping + balanced flood / non-flood points
- factor_stack():    the 14 conditioning factors as one multi-band image
- export_*():        send results to Google Drive
"""
import ee

from .config import CRS, DRIVE_FOLDER, EE_PROJECT, GEE_COLUMNS, SCENARIO

FLOOD_CFG = {
    'scale': 30,              # output resolution (m)
    'speckle_radius': 50,     # focal median radius (m)
    'diff_db': -3,            # post − pre must drop by at least 3 dB
    'thr_min': -22, 'thr_max': -15,   # accepted Otsu range (dB)
    'thr_default': -18,       # used when Otsu falls outside that range
    'max_slope': 5,           # degrees
    'max_hand': 20,           # m above nearest drainage
    'perm_months': 10,        # JRC seasonality >= this = permanent water
    'min_pixels': 8,          # remove flood patches smaller than this
    'nonflood_buffer': 500,   # non-flood points must be >= 500 m from flood
    'points_per_class': 1500,
    'seed': 42,
}

FACTOR_CFG = {'rain_days': 5, 'antec_days': 30, 'stream_upa_km2': 10}


def init(project=EE_PROJECT):
    """Connect to Earth Engine (opens a browser login the first time)."""
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
    print('Earth Engine ready — project:', project)


def rect(box):
    return ee.Geometry.Rectangle(box)


# ---------------------------------------------------------------- datasets

def s1_vv():
    return (ee.ImageCollection('COPERNICUS/S1_GRD')
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .select('VV'))


def permanent_water(months=FLOOD_CFG['perm_months']):
    return (ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
            .select('seasonality').unmask(0).gte(months))


def dem():
    col = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM')
    return col.mosaic().setDefaultProjection(col.first().projection())


# ---------------------------------------------------------------- flood mapping

def _describe(col, aoi):
    def add(img):
        cov = (img.geometry().intersection(aoi, 1000).area(1000)
               .divide(aoi.area(1000)).multiply(100))
        return img.set({'date': img.date().format('YYYY-MM-dd'), 'coverage_pct': cov})
    return col.map(add)


def otsu(histogram):
    """Otsu threshold from an ee.Reducer.histogram() result (last bucket skipped)."""
    counts = ee.Array(ee.Dictionary(histogram).get('histogram'))
    means = ee.Array(ee.Dictionary(histogram).get('bucketMeans'))
    size = means.length().get([0])
    total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
    total_sum = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
    mean = total_sum.divide(total)

    def between_class_variance(i):
        a_counts = counts.slice(0, 0, i)
        a_count = a_counts.reduce(ee.Reducer.sum(), [0]).get([0])
        a_mean = (means.slice(0, 0, i).multiply(a_counts)
                  .reduce(ee.Reducer.sum(), [0]).get([0]).divide(a_count))
        b_count = total.subtract(a_count)
        b_mean = total_sum.subtract(a_count.multiply(a_mean)).divide(b_count)
        return (a_count.multiply(a_mean.subtract(mean).pow(2))
                .add(b_count.multiply(b_mean.subtract(mean).pow(2))))

    bss = ee.List.sequence(1, size.subtract(1)).map(between_class_variance)
    return means.slice(0, 0, size.subtract(1)).sort(bss).get([-1])


def flood_inventory(ev, cfg=FLOOD_CFG):
    """Map one flood event and sample balanced flood / non-flood points.

    Returns a dict of ee objects; call .getInfo() on the 'info' values to see them.
    """
    aoi = rect(ev['aoi'])
    s1 = s1_vv()
    post_all = _describe(s1.filterBounds(aoi).filterDate(*ev['post']), aoi)
    pre_all = _describe(s1.filterBounds(aoi).filterDate(*ev['pre']), aoi)

    # Best-covering flood image whose orbit also exists in the pre-flood window
    def count_pre(img):
        n = (pre_all
             .filter(ee.Filter.eq('relativeOrbitNumber_start', img.get('relativeOrbitNumber_start')))
             .filter(ee.Filter.eq('orbitProperties_pass', img.get('orbitProperties_pass')))
             .size())
        return img.set('n_pre', n)
    post_matched = post_all.map(count_pre).filter(ee.Filter.gt('n_pre', 0))
    best = ee.Image(post_matched.sort('coverage_pct', False).first())
    same_orbit = ee.Filter.And(
        ee.Filter.eq('relativeOrbitNumber_start', best.get('relativeOrbitNumber_start')),
        ee.Filter.eq('orbitProperties_pass', best.get('orbitProperties_pass')))
    post_col, pre_col = post_all.filter(same_orbit), pre_all.filter(same_orbit)
    proj = best.projection()

    r = cfg['speckle_radius']
    pre = (pre_col.median().setDefaultProjection(proj)
           .focal_median(r, 'circle', 'meters').clip(aoi))
    post = (post_col.sort('system:time_start', False).mosaic().setDefaultProjection(proj)
            .focal_median(r, 'circle', 'meters').clip(aoi))
    valid = pre.mask().And(post.mask())

    perm = permanent_water(cfg['perm_months'])
    slope = ee.Terrain.slope(dem())
    hand = ee.Image('MERIT/Hydro/v1_0_1').select('hnd')

    # Water threshold: Otsu near permanent water, fallback when implausible
    near_water = perm.focal_max(radius=3000, units='meters')
    hist = post.updateMask(near_water).reduceRegion(
        reducer=ee.Reducer.histogram(255, 0.1), geometry=aoi, scale=90,
        maxPixels=1e10, bestEffort=True).get('VV')
    raw_thr = ee.Number(otsu(hist))
    otsu_ok = raw_thr.gte(cfg['thr_min']).And(raw_thr.lte(cfg['thr_max']))
    thr = ee.Number(ee.Algorithms.If(otsu_ok, raw_thr, cfg['thr_default']))

    flood_raw = (post.lt(thr)
                 .And(post.subtract(pre).lt(cfg['diff_db']))
                 .And(perm.Not())
                 .And(slope.lt(cfg['max_slope']))
                 .And(hand.lt(cfg['max_hand']))
                 .reproject(crs=CRS, scale=cfg['scale']))
    connected = flood_raw.selfMask().connectedPixelCount(cfg['min_pixels'] + 1, True)
    flood = (connected.gte(cfg['min_pixels']).unmask(0)
             .updateMask(valid).clip(aoi).rename('flood').toByte())

    area_km2 = ee.Number(flood.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=cfg['scale'], crs=CRS,
        maxPixels=1e13, bestEffort=True).get('flood')).divide(1e6)

    # 1 = flood, 0 = non-flood at least nonflood_buffer m away; permanent water excluded
    near_flood = flood.focal_max(radius=cfg['nonflood_buffer'], units='meters')
    label = flood.where(flood.eq(0).And(near_flood.eq(1)), 2).updateMask(perm.Not())
    label = label.updateMask(label.neq(2)).rename('label').toInt()

    n = cfg['points_per_class']
    name = ev['name']

    def add_props(f):
        c = f.geometry().coordinates()
        return f.set({'event': name, 'lon': c.get(0), 'lat': c.get(1)})

    points = label.stratifiedSample(
        numPoints=0, classBand='label', region=aoi, scale=cfg['scale'], projection=CRS,
        seed=cfg['seed'], classValues=[0, 1], classPoints=[n, n],
        geometries=True, tileScale=4).map(add_props)

    return {
        'aoi': aoi, 'pre': pre, 'post': post, 'flood': flood, 'points': points,
        'permanent_water': perm,
        'info': {
            'post_dates': post_matched.aggregate_array('date'),
            'orbit': best.get('relativeOrbitNumber_start'),
            'pass': best.get('orbitProperties_pass'),
            'n_pre_images': pre_col.size(),
            'n_post_images': post_col.size(),
            'otsu_raw_db': raw_thr,
            'threshold_used_db': thr,
            'flooded_area_km2': area_km2,
        },
    }


# ---------------------------------------------------------------- factors

def static_factors(stream_upa_km2=FACTOR_CFG['stream_upa_km2']):
    """Factors that do not change between events."""
    d = dem()
    slope = ee.Terrain.slope(d)
    merit = ee.Image('MERIT/Hydro/v1_0_1')
    upa = merit.select('upa')                         # upstream area (km²)
    tan_b = slope.multiply(3.141592653589793 / 180).tan().max(0.001)
    sca = upa.multiply(1e6).divide(90)                # specific catchment area (m²/m)

    streams = upa.gt(stream_upa_km2).unmask(0).reproject(crs=CRS, scale=90)
    dist_river = (streams.fastDistanceTransform(256).sqrt().multiply(90)
                  .unmask(256 * 90).rename('dist_river'))
    # Share of 90 m stream pixels within 1 km × (0.09 km of channel per 0.0081 km² pixel)
    drain_dens = (streams.focal_mean(radius=1000, kernelType='circle', units='meters')
                  .reproject(crs=CRS, scale=90).divide(0.09).rename('drain_dens'))

    return ee.Image.cat([
        d.rename('elevation'),
        slope.rename('slope'),
        ee.Terrain.aspect(d).rename('aspect'),
        d.convolve(ee.Kernel.laplacian8(normalize=False)).rename('curvature'),
        sca.divide(tan_b).log().rename('twi'),
        sca.multiply(tan_b).add(1).log().rename('spi'),
        merit.select('hnd').rename('hand'),
        dist_river,
        drain_dens,
        ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').rename('lulc'),
        ee.Image('OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02').select('b0').rename('soil_tex'),
    ])


def _mask_s2(img):
    scl = img.select('SCL')
    clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return img.updateMask(clear)


def ndvi_before(date, region):
    d = ee.Date(date)
    return (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(region).filterDate(d.advance(-1, 'year'), d)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
            .map(_mask_s2).median()
            .normalizedDifference(['B8', 'B4']).rename('ndvi'))


def event_rain(date, cfg=FACTOR_CFG):
    d = ee.Date(date)
    start = d.advance(-cfg['rain_days'], 'day')
    rain_event = (ee.ImageCollection('NASA/GPM_L3/IMERG_V07').filterDate(start, d)
                  .select('precipitation').sum().multiply(0.5).rename('rain_event'))
    rain_antec = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                  .filterDate(start.advance(-cfg['antec_days'], 'day'), start)
                  .select('precipitation').sum().rename('rain_antec'))
    return ee.Image.cat([rain_event, rain_antec])


def factor_stack(region, flood_date=None, scenario=SCENARIO):
    """All 14 factors (GEE_COLUMNS order). Real event rainfall if flood_date is given,
    otherwise a constant rainfall scenario."""
    if flood_date:
        dynamic = ndvi_before(flood_date, region).addBands(event_rain(flood_date))
    else:
        rain = (ee.Image.constant([scenario['rain_event'], scenario['rain_antec']])
                .rename(['rain_event', 'rain_antec']).toFloat())
        dynamic = ndvi_before(scenario['ndvi_date'], region).addBands(rain)
    return static_factors().addBands(dynamic).select(GEE_COLUMNS)


def sample_factors(points, flood_date):
    """Attach factor values to every point (points without full data are dropped)."""
    stack = factor_stack(points.geometry().bounds(), flood_date)
    return stack.sampleRegions(collection=points,
                               properties=['label', 'event', 'lon', 'lat'],
                               scale=30, projection=CRS, tileScale=8, geometries=False)


# ---------------------------------------------------------------- exports

def export_image(image, name, region, scale=30, folder=DRIVE_FOLDER):
    task = ee.batch.Export.image.toDrive(image=image, description=name, folder=folder,
                                         region=region, scale=scale, crs=CRS,
                                         maxPixels=1e13)
    task.start()
    print('Started export:', name)
    return task


def export_table(fc, name, file_format='CSV', selectors=None, folder=DRIVE_FOLDER):
    task = ee.batch.Export.table.toDrive(collection=fc, description=name, folder=folder,
                                         fileFormat=file_format, selectors=selectors)
    task.start()
    print('Started export:', name)
    return task

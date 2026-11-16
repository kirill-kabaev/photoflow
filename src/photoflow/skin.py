"""Conservative face-only chroma-field correction; no luminance smoothing."""
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

from .storage import FlowError


def natural_skin_retouch(image, skin, settings, spatial_scale=1.0):
    """Modify developed linear RGB in place, preserving luminance and fine detail.

    Gaussian filters estimate ONLY low-frequency colour correction fields; their
    output never replaces skin pixels. Dark marks/strong edges are protected.
    This is deterministic colour retouching, not blemish classification.
    """
    if (image.dtype != np.float32 or image.ndim != 3 or image.shape[2] != 3
            or skin.shape != image.shape[:2] or not np.isfinite(skin).all() or not np.isfinite(image).all()
            or skin.min()<0 or skin.max()>1 or not 0 < spatial_scale <= 1):
        raise FlowError('Invalid natural skin input/mask/scale')
    if not settings.enabled or (settings.tone_evenness == 0 and settings.redness == 0):
        return image
    ys, xs = np.where(skin > 0)
    if not len(xs):
        return image
    x1,x2,y1,y2 = xs.min(),xs.max()+1,ys.min(),ys.max()+1
    face, mask = image[y1:y2,x1:x2], skin[y1:y2,x1:x2]
    if min(face.shape[:2])<3:
        return image
    weights = np.array([.2126,.7152,.0722],np.float32)
    lum = face @ weights
    chroma = face-lum[...,None]
    def colour_field(sigma):
        denominator = gaussian_filter(mask,sigma)
        return np.stack([gaussian_filter(chroma[...,c]*mask,sigma)/np.maximum(denominator,1e-6)
                         for c in range(3)],axis=-1)
    low = colour_field(max(.5,8*spatial_scale))
    broad = colour_field(max(1,28*spatial_scale))
    delta = settings.tone_evenness * np.clip(broad-low,-.04,.04)
    red_excess = np.maximum(low[...,0]-low[...,1],0)*settings.redness
    delta[...,0] -= red_excess
    delta[...,1] += red_excess*(weights[0]/weights[1])
    # Remove floating point luminance drift and cap the total colour change.
    delta -= (delta@weights)[...,None]
    magnitude = np.max(np.abs(delta),axis=-1)
    delta *= np.minimum(1,.012/np.maximum(magnitude,1e-8))[...,None]
    gy,gx = np.gradient(lum)
    permanent = (lum-gaussian_filter(lum,max(.5,3*spatial_scale)) < -.02) | (np.hypot(gx,gy)>.035)
    protected = binary_dilation(permanent,iterations=max(1,round(2*spatial_scale)))
    inside = binary_erosion(mask>.9,iterations=max(1,round(2*spatial_scale)))
    alpha = mask*inside*(~protected)
    change = delta*alpha[...,None]
    # Reduce rather than clip an RGB channel, retaining luminance at gamut edges.
    permitted = np.min(np.where(change<0,face/np.maximum(-change,1e-8),1),axis=-1)
    face += change*np.minimum(1,np.maximum(permitted,0))[...,None]
    return image


def preview_existing(workflow, identifier, version=None):
    """Isolated native crop demonstration; does not reserve/render a RAW version.

    Input is an existing RAW-derived native JPEG. Explicitly not a new RAW grade.
    Full processing uses the same operation before JPEG encoding on linear RAW.
    """
    import hashlib
    import json
    from PIL import Image
    from . import imaging
    from .models import NaturalSkinRetouch
    from .segmentation import SemanticCache
    from .storage import digest, json_bytes, write_json
    photo = workflow.photo(identifier)
    version = version or photo.current_version
    root, recipe, _ = workflow.validate_version(photo,version)
    if not recipe.segmentation or not any(r.id == 'face' for r in recipe.detail_regions):
        raise FlowError('Skin preview requires existing segmentation and native face crop')
    cache = SemanticCache(workflow.session,photo,recipe.segmentation.cache_key)
    region = json.loads((root/'details/face/region.json').read_text())
    x1,y1,x2,y2 = region['source_box']
    source = root/'details/face/after.jpg'
    with Image.open(source) as image:
        encoded = np.asarray(image.convert('RGB'),dtype=np.float32)/255
    if region['scale'] != 1 or encoded.shape[:2] != (y2-y1,x2-x1):
        raise FlowError('Skin preview requires native 100% source crop')
    full = cache.mask('face_skin',(cache.manifest.dimensions[1],cache.manifest.dimensions[0],3))
    mask = full[y1:y2,x1:x2].copy(); del full
    before = imaging.linearize_srgb(encoded)
    after = natural_skin_retouch(before.copy(),mask,NaturalSkinRetouch())
    record = {'input_sha256':digest(source),'version':version,'source_box':region['source_box'],
        'cache_key':cache.key,'natural_skin':NaturalSkinRetouch().model_dump(), 'scale':1.0,
        'source':'existing native RAW-derived JPEG, decoded to linear sRGB; preview-only, not a new RAW version',
        'spot_healing':False,'full_passes_used':0}
    name = hashlib.sha256(json_bytes(record)).hexdigest()[:32]
    folder = workflow.session.work(photo.id,'skin_previews',name)
    if not folder.exists():
        folder.mkdir(parents=True)
        imaging.save_previews(before,after,folder,max_edge=max(after.shape[:2]))
        record['max_linear_luminance_change'] = float(np.max(np.abs(imaging.luminance(before)-imaging.luminance(after))))
        record['artifacts'] = {p.name:digest(p) for p in folder.iterdir()}
        write_json(folder/'manifest.json',record)
    return {'photo_id':photo.id,'version':version,'before':str(folder/'before.jpg'),
        'after':str(folder/'after.jpg'),'comparison':str(folder/'before_after.jpg'),
        'manifest':str(folder/'manifest.json'),'note':record['source']}

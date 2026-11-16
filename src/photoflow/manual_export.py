"""Explicit JPEG delivery; RAW recipe replay requires a separate opt-in."""
import json
import os
import uuid

import numpy as np
from PIL import Image

from . import imaging
from .storage import FlowError, digest, now, write_json, sync_dir


def export_saved(workflow, identifier, version, actor, reason, render_saved_recipe=False):
    """Caller holds Session.lock. Permission to deliver is not artistic approval."""
    if not version or not actor or not actor.strip() or not reason or not reason.strip():
        raise FlowError('Manual export requires --force --version --by --reason')
    photo=workflow.photo(identifier)
    workflow.session.source(photo)
    folder,recipe,manifest=workflow.validate_version(photo,version)
    filename=f'{photo.id}.jpg'
    destination=workflow.session.path('final',filename)
    record=workflow.session.path('final',f'{photo.id}.manual-export.json')
    if destination.exists() or record.exists():
        raise FlowError('Final file already exists; refusing overwrite')
    if 'full_result.npy' not in manifest['artifacts'] and not render_saved_recipe:
        raise FlowError(f'{photo.filename} {version}: saved full-resolution processed result unavailable; RAW reprocessing is prohibited, preview export refused')
    staging=workflow.session.work(photo.id,'manual_exports',uuid.uuid4().hex)
    staging.mkdir(parents=True)
    rendered=False
    if 'full_result.npy' in manifest['artifacts']:
        pixels=np.load(folder/'full_result.npy',mmap_mode='r',allow_pickle=False)
        pixels_checksum=manifest['artifacts']['full_result.npy']
    else:
        if manifest['libraries']!=imaging.runtime_versions():
            raise FlowError('Rendering libraries changed; cannot replay saved version unchanged')
        workflow.validate_crop(photo,recipe)
        semantics=workflow.semantic_cache(photo,recipe)  # Cache only, never inference.
        healing=workflow.healing_provider(photo,recipe)
        decoded=workflow.decoder(workflow.session.source(photo),recipe.raw)
        if decoded.metadata['oriented_dimensions']!=json.loads((folder/'decoder.json').read_text())['oriented_dimensions']:
            raise FlowError('RAW orientation/dimensions differ from saved version')
        pixels=imaging.process_pixels(decoded.pixels,recipe,semantic_provider=semantics,healing_provider=healing)
        np.save(staging/'full_result.npy',pixels,allow_pickle=False)
        pixels_checksum=digest(staging/'full_result.npy')
        rendered=True
    dimensions=json.loads((folder/'decoder.json').read_text())['oriented_dimensions']
    x1,y1,x2,y2=imaging.crop_box((dimensions[1],dimensions[0],3),recipe.crop,recipe.approved_crop)
    if (pixels.dtype!=np.float32 or pixels.shape!=(y2-y1,x2-x1,3) or not np.isfinite(pixels).all()):
        raise FlowError('Saved developed pixels have invalid native dimensions/dtype/values')
    jpeg=staging/filename
    Image.fromarray(np.uint8(np.rint(imaging.encode_srgb(pixels)*255))).save(
        jpeg,quality=recipe.output.jpeg_quality,subsampling=0,icc_profile=imaging.srgb_profile())
    with Image.open(jpeg) as test:
        test.load()
        if test.size!=(x2-x1,y2-y1) or test.mode!='RGB':
            raise FlowError('Manual JPEG dimensions or mode invalid')
        imaging.validate_srgb_profile(test.info.get('icc_profile',b''))
    workflow.session.source(photo)
    metadata={'mode':'explicit_manual_export','artistically_approved':False,
        'authorized_by':actor,'reason':reason,'created_at':now(),'photo_id':photo.id,
        'source_version':version,'source_sha256':photo.source_sha256,
        'recipe_sha256':manifest['recipe_sha256'],'processed_pixels_sha256':pixels_checksum,
        'dimensions':[x2-x1,y2-y1],'color_space':'sRGB','jpeg_quality':recipe.output.jpeg_quality,
        'files':{filename:digest(jpeg)},'raw_reprocessed':rendered}
    write_json(staging/'manifest.json',metadata)
    linked=[]
    try:
        for source,target in [(jpeg,destination),(staging/'manifest.json',record)]:
            os.link(source,target); linked.append(target)
        sync_dir(destination.parent)
    except Exception:
        for path in linked:
            path.unlink()
        raise
    return {'photo_id':photo.id,'version':version,'files':[str(destination)],'artistically_approved':False}

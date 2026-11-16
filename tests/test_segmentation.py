"""Deterministic fixtures validate engineering, never real neural model quality."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from pydantic import ValidationError

from conftest import make_plan, process_first
from photoflow import imaging
from photoflow.models import Plan, Recipe, SemanticRegion
from photoflow.segmentation import (Letterbox, SegmentConfig, SemanticCache, LocalParser,
                                    face_roll, region_support, refine_mask, segment)
from photoflow.storage import FlowError, digest, write_json


class FixtureParser:
    identity = {'id':'explicit_test_fixture_not_neural','weights':{'fixture_sha':'1'*64}}
    execution = {'adapter':'deterministic_test_fixture_not_real_inference'}
    timings = {}
    calls = 0

    def predict(self,image):
        self.calls += 1
        body = np.zeros((image.height,image.width),np.uint8)
        body[5:15,25:45] = 2
        body[15:30,25:45] = 11
        body[30:55,15:70] = 4
        body[55:75,15:70] = 6
        body[35:45,70:78] = 14
        face = np.zeros_like(body)
        face[15:30,25:45] = 1
        face[18:20,30:33] = 4
        face[25:27,32:36] = 13
        return body,face,{'fixture':True}


def cached(workflow,config=None):
    fixture = FixtureParser()
    photo = workflow.queue.photos[0]
    result = segment(workflow,photo.id,config=config,parser=fixture)
    return SemanticCache(workflow.session,photo,result['cache_key']),fixture


def semantic_plan(workflow,cache):
    photo = workflow.queue.photos[0]
    workflow.prepare(photo.id)
    path = make_plan(workflow,photo)
    data = json.loads(path.read_text())
    data['recipe']['segmentation'] = {'cache_key':cache.key}
    data['recipe']['semantic_masks'] = [{'id':'face_only','region':'face_skin','description':'Test fixture facial mask'}]
    data['recipe']['local_adjustments'] = [{'mask':'face_only','operation':'exposure','amount':0.2,'rationale':'Synthetic test correction'}]
    write_json(path,data)
    return path


def test_label_mapping_excludes_hair_lips_eyes_and_keeps_dress_separate():
    body = np.array([[0,2,4,5,6,7,11,11,11,11,14,15]],np.uint8)
    face = np.array([[0,0,0,0,0,0,1,4,13,10,0,0]],np.uint8)
    assert region_support(body,face,'face_skin').tolist()==[[False]*6+[True,False,False,True,False,False]]
    assert np.where(region_support(body,face,'lower_clothing'))[1].tolist()==[3,4]
    assert np.where(region_support(body,face,'upper_clothing'))[1].tolist()==[2]
    assert np.where(region_support(body,face,'skin'))[1].tolist()==[6,9,10,11]
    assert np.array_equal(~region_support(body,face,'person'),region_support(body,face,'background'))
    with pytest.raises(FlowError,match='Unsupported'):
        region_support(body,face,'imaginary')


@pytest.mark.parametrize('width,height',[(300,600),(600,300),(311,93),(93,311),(512,512)])
def test_letterbox_alignment_padding_aspect_ratio_and_no_flips(width,height):
    transform = Letterbox.create(width,height)
    assert abs(transform.resized_width/transform.resized_height-width/height)<0.02
    image = np.zeros((height,width,3),np.uint8)
    image[:height//2,:width//2] = [255,0,0]
    canvas = transform.apply(Image.fromarray(image))
    labels = (canvas[...,0]>0.5).astype(np.uint8)*3
    restored = transform.restore(labels)
    assert restored.shape==(height,width) and set(np.unique(restored))<={0,3}
    assert restored[height//4,width//4]==3
    assert restored[3*height//4,width//4]==0
    assert restored[height//4,3*width//4]==0
    assert np.mean(restored[:height//2,:width//2]==3)>0.97


def test_letterbox_restore_handles_smaller_neural_output():
    t = Letterbox.create(120,80)
    labels = np.zeros((128,128),np.uint8);labels[45:70,45:70]=11
    restored = t.restore(labels)
    assert restored.shape==(80,120) and set(np.unique(restored))=={0,11}
    assert restored[40,60]==11


def test_roll_estimation_and_inverse_rotation_preserve_coordinates():
    y,x=np.mgrid[:120,:120]
    angle=np.radians(35)
    dx,dy=x-60,y-60
    support=((dx*np.cos(angle)-dy*np.sin(angle))/15)**2+((dx*np.sin(angle)+dy*np.cos(angle))/35)**2<1
    yy,xx=np.where(support)
    roll=face_roll(xx,yy)
    assert abs(roll-35)<1
    original=Image.fromarray(support.astype(np.uint8))
    returned=original.rotate(-roll,Image.Resampling.NEAREST).rotate(roll,Image.Resampling.NEAREST)
    a,b=np.asarray(original)>0,np.asarray(returned)>0
    assert (a&b).sum()/(a|b).sum()>0.97


def test_guided_refinement_is_finite_bounded_and_has_no_outward_spill():
    support=np.zeros((80,120),bool);support[20:65,30:95]=True
    guide=support.astype(np.float32)*0.5+0.1
    refined=refine_mask(support,guide,SegmentConfig())
    assert refined.dtype==np.float32 and np.isfinite(refined).all()
    assert 0<=refined.min()<=refined.max()<=1
    assert np.all(refined[~support]==0)
    assert refined[40,60]>0.99 and refined[20,30]<1
    assert np.array_equal(refine_mask(support,guide,SegmentConfig(refine=False)),support.astype(np.float32))


def test_semantic_cache_runs_once_and_reuses_for_revisions_and_resizing(workflow):
    cache,parser=cached(workflow)
    second=segment(workflow,workflow.queue.photos[0].id,parser=parser)
    assert second['cache_hit'] and parser.calls==1
    assert cache.manifest.execution['adapter']=='deterministic_test_fixture_not_real_inference'
    assert cache.manifest.dimensions==[120,80]
    full=cache.mask('upper_clothing',(160,240,3))
    assert full.shape==(160,240) and full.dtype==np.float32 and full.min()>=0 and full.max()<=1
    assert full[80,80]>0.9 and full[5,5]==0
    assert np.all(full[:10]==0)
    assert not list(cache.folder.glob('.staging*'))


@pytest.mark.parametrize('change',['orientation','refinement','weights'])
def test_cache_invalidates_when_relevant_inputs_change(workflow,change):
    cache,parser=cached(workflow)
    config=SegmentConfig()
    if change=='refinement':
        config=config.model_copy(update={'refine':False})
    elif change=='weights':
        parser.identity={'id':'fixture-v2','weights':{'sha':'2'*64}}
    else:
        original=workflow.decoder
        def rotated(path,settings):
            decoded=original(path,settings);decoded.metadata['orientation_flip']=6
            return decoded
        workflow.decoder=rotated
    result=segment(workflow,workflow.queue.photos[0].id,config=config,parser=parser)
    assert result['cache_key']!=cache.key and not result['cache_hit']
    assert parser.calls==2


def test_inference_failure_never_publishes_partial_cache_or_fake_masks(workflow):
    class Broken(FixtureParser):
        def predict(self,image):
            raise FlowError('Synthetic model failure')
    photo=workflow.queue.photos[0]
    with pytest.raises(FlowError,match='model failure'):
        segment(workflow,photo.id,parser=Broken())
    root=workflow.session.work(photo.id,'segmentation')
    assert list(root.glob('.staging-*'))
    assert not list(root.glob('*/manifest.json'))
    assert photo.render_attempts==[]


def test_missing_and_corrupt_masks_fail_without_ellipse_fallback(workflow):
    cache,_=cached(workflow)
    with pytest.raises(FlowError,match='unavailable'):
        cache.mask('dress',(80,120,3))
    (cache.folder/'hair.npy').write_bytes(b'corrupt')
    with pytest.raises(FlowError,match='artifact changed'):
        SemanticCache(workflow.session,workflow.queue.photos[0],cache.key)


def test_missing_model_installation_is_explicit(tmp_path):
    with pytest.raises(FlowError,match='models missing'):
        LocalParser(tmp_path)


def test_recipe_rejects_unknown_semantics_missing_binding_and_bad_intersection(workflow):
    cache,_=cached(workflow);path=semantic_plan(workflow,cache)
    data=json.loads(path.read_text())
    for patch in [{'segmentation':None},{'semantic_masks':[{'id':'bad','region':'fake_skin','description':'fixture'}]},
                  {'semantic_masks':[{'id':'face_only','region':'face_skin','description':'fixture','intersect_ellipse':'missing'}]}]:
        bad={**data,'recipe':{**data['recipe'],**patch}}
        with pytest.raises(ValidationError):
            Plan.model_validate_json(json.dumps(bad))


@pytest.mark.parametrize('operation,amount',[('exposure',0.3),('saturation',-0.1),('redness',0.1),('sharpen',0.1),('smooth',0.1)])
def test_blending_preserves_all_unmasked_pixels_and_applies_before_crop(workflow,pixels,operation,amount):
    cache,_=cached(workflow);path=semantic_plan(workflow,cache)
    recipe=Plan.model_validate_json(path.read_bytes()).recipe
    recipe.global_adjustments.exposure_ev=0.
    recipe.local_adjustments[0].operation=operation;recipe.local_adjustments[0].amount=amount
    reference=recipe.model_copy(deep=True);reference.semantic_masks=[];reference.segmentation=None;reference.local_adjustments=[]
    neutral=imaging.process_pixels(pixels,reference)
    result=imaging.process_pixels(pixels,recipe,semantic_provider=cache)
    mask=cache.mask('face_skin',pixels.shape)
    assert np.array_equal(result[mask==0],neutral[mask==0])
    recipe.crop.left=0.1;recipe.crop.top=0.1;recipe.crop.right=0.9;recipe.crop.bottom=0.9
    assert np.array_equal(imaging.process_pixels(pixels,recipe,semantic_provider=cache),imaging.crop_image(result,recipe.crop))
    with pytest.raises(FlowError,match='require'):
        imaging.process_pixels(pixels,recipe)


def test_ellipse_intersection_semantic_version_provenance_and_approval_gate(workflow):
    cache,_=cached(workflow);path=semantic_plan(workflow,cache)
    data=json.loads(path.read_text())
    data['recipe']['masks']=[{'id':'light','center':[0.3,0.27],'radius':[0.2,0.2],'feather':0.5,'description':'Synthetic lighting gradient'}]
    data['recipe']['semantic_masks'][0]['intersect_ellipse']='light'
    write_json(path,data)
    workflow.import_plan(workflow.queue.photos[0].id,path)
    result=workflow.process(workflow.queue.photos[0].id)
    photo=workflow.queue.photos[0]
    folder,recipe,manifest=workflow.validate_version(photo,result['version'])
    assert (folder/'segmentation.json').is_file()
    assert (folder/'masks/face_only-overlay.jpg').is_file()
    assert not (folder/'masks/face_only.npy').exists()  # Reconstruct from shared cached alpha, no duplicated full arrays.
    assert (folder/'masks/light.npy').is_file()
    assert manifest['segmentation']['cache_key']==cache.key
    assert 'semantic face_skin' in (folder/'report.md').read_text()
    with pytest.raises(FlowError,match='approval|approved'):
        workflow.export(photo.id)
    assert not list(workflow.session.path('final').iterdir())


def test_explicit_new_user_cycle_keeps_prior_attempts_and_has_own_hard_limit(workflow):
    photo=process_first(workflow)
    previous=photo.render_attempts[0].model_dump()
    workflow.begin_cycle(photo.id,'User via explicit task','Stage 2 new feature validation',max_passes=1)
    assert len(photo.render_attempts)==1 and photo.active_attempt_count==0
    assert photo.render_attempts[0].model_dump()==previous
    with pytest.raises(FlowError,match='previous processing cycle'):
        workflow.begin_cycle(photo.id,'invalid','Cannot silently reset an unused cycle')
    workflow.request_revision(photo.id)
    path=make_plan(workflow,photo,parent='v001',feedback=workflow.feedback_path(photo))
    workflow.import_plan(photo.id,path);workflow.process(photo.id)
    assert len(photo.render_attempts)==2 and photo.active_attempt_count==1
    workflow.request_revision(photo.id)
    path=make_plan(workflow,photo,parent='v002',feedback=workflow.feedback_path(photo))
    workflow.import_plan(photo.id,path)
    with pytest.raises(FlowError,match='pass limit'):
        workflow.process(photo.id)
    assert photo.render_attempts[0].model_dump()==previous


def test_director_semantic_context_and_preview_use_cached_masks(workflow,monkeypatch):
    from photoflow.director import Director,DirectorConfig,ANALYSIS_CATEGORIES
    import photoflow.segmentation as segmentation
    cache,fixture=cached(workflow)
    actual_segment=segmentation.segment
    monkeypatch.setattr(segmentation,'segment',lambda wf,p:actual_segment(wf,p,parser=fixture))
    director=Director(workflow)
    action=director.start('Synthetic semantic workflow',workflow.queue.photos[0].id,DirectorConfig(semantic=True))
    assert action['semantic_segmentation']['cache_key']==cache.key
    path=make_plan(workflow,workflow.queue.photos[0])
    data=json.loads(path.read_text())
    data['analysis']['coverage']={key:'Explicit synthetic test observation' for key in ANALYSIS_CATEGORIES}
    data['recipe'].update(style='Natural Commercial Portrait',segmentation={'cache_key':cache.key},
                          semantic_masks=[{'id':'hair_only','region':'hair','description':'Fixture region'}],
                          local_adjustments=[{'mask':'hair_only','operation':'exposure','amount':0.1,'rationale':'fixture'}])
    write_json(path,data)
    result=director.explore(workflow.queue.photos[0].id,path,'semantics-fixture')
    assert result['full_passes_used']==0 and fixture.calls==1
    assert Path(result['preview']).exists()


def test_model_output_failure_is_not_replaced_with_ellipses(workflow):
    class Invalid(FixtureParser):
        def predict(self,image):
            body,face,transforms=super().predict(image)
            return body.astype(np.float32),face,transforms
    with pytest.raises(FlowError,match='invalid labels'):
        segment(workflow,workflow.queue.photos[0].id,parser=Invalid())
    assert workflow.queue.photos[0].render_attempts==[]


def test_cpu_provider_fallback_and_model_checksum_validation(tmp_path,monkeypatch):
    import sys
    import types
    from photoflow.segmentation import _ort_session
    model=tmp_path/'fixture.onnx';model.write_bytes(b'explicitly not a real ONNX model')
    chosen=[]
    class Options:
        pass
    def new_session(path,sess_options,providers):
        chosen.append(providers)
        if 'CUDAExecutionProvider' in providers:
            raise RuntimeError('Synthetic unavailable CUDA libraries')
        return 'cpu fixture session'
    monkeypatch.setitem(sys.modules,'onnxruntime',types.SimpleNamespace(
        get_available_providers=lambda:['CUDAExecutionProvider','CPUExecutionProvider'],
        SessionOptions=Options,InferenceSession=new_session))
    _ort_session.cache_clear()
    assert _ort_session(str(model),digest(model),'auto')=='cpu fixture session'
    assert chosen==[['CUDAExecutionProvider','CPUExecutionProvider'],['CPUExecutionProvider']]
    _ort_session.cache_clear()
    with pytest.raises(FlowError,match='checksum'):
        _ort_session(str(model),'0'*64,'cpu')


@pytest.mark.raw
def test_real_local_parsing_of_raw_separately_from_fixtures():
    import os
    model_dir=os.environ.get('PHOTOFLOW_SEGMENTATION_MODELS')
    sample=os.environ.get('PHOTOFLOW_RAW_SAMPLE')
    if not model_dir or not sample:
        pytest.skip('Set RAW sample and local model directory for actual neural inference')
    from photoflow.segmentation import _preview
    from photoflow.models import RawSettings
    path=Path(sample)
    original=digest(path)
    decoded=imaging.decode_raw(path,RawSettings())
    preview=_preview(decoded,SegmentConfig())
    parser=LocalParser(model_dir,device='cpu')
    body,face,transforms=parser.predict(preview)
    assert body.shape==(preview.height,preview.width)
    assert parser.execution['body_providers']==['CPUExecutionProvider']
    assert {0,2,4,6,11}<=set(np.unique(body))
    assert region_support(body,face,'face_skin').sum()>0
    assert region_support(body,face,'eyes').sum()>0
    assert region_support(body,face,'lips').sum()>0
    assert transforms['face_crops']
    assert digest(path)==original


def test_real_artifact_mask_diagnostics_have_native_bounds_and_protect_history(workflow):
    from photoflow.segmentation import boundary_diagnostics,compare_masks
    cache,_=cached(workflow);path=semantic_plan(workflow,cache)
    data=json.loads(path.read_text())
    data['recipe']['masks']=[{'id':'lighting','center':[0.3,0.27],'radius':[0.3,0.3],'description':'Synthetic comparison ellipse'}]
    data['recipe']['detail_regions']=[{'id':'face','bounds':{'left':0.1,'top':0.1,'right':0.5,'bottom':0.5},'description':'Synthetic native detail region'}]
    write_json(path,data)
    photo=workflow.queue.photos[0]
    workflow.import_plan(photo.id,path);workflow.process(photo.id)
    folder=workflow.session.work(photo.id,'v001')
    before={str(p.relative_to(folder)):digest(p) for p in folder.rglob('*') if p.is_file()}
    result=boundary_diagnostics(workflow,photo.id,'v001',cache.key)
    assert result['details']['face_skin-face.jpg']['source_box']==[12,8,60,40]
    assert result['details']['face_skin-face.jpg']['dimensions']==[48,32]
    assert Path(result['labels']).is_file()
    comparison=compare_masks(workflow,photo.id,'v001',cache.key,{'face_skin':['lighting']})
    assert Path(comparison['comparison']).is_file()
    assert 0<=comparison['measurements']['face_skin']['ellipse_alpha_outside_predicted_semantic_fraction']<=1
    after={str(p.relative_to(folder)):digest(p) for p in folder.rglob('*') if p.is_file()}
    assert before==after
    with pytest.raises(FlowError,match='refusing overwrite'):
        boundary_diagnostics(workflow,photo.id,'v001',cache.key)


def test_atomic_cache_write_failure_preserves_staging_without_publication(workflow,monkeypatch):
    import photoflow.segmentation as segmentation
    save=segmentation._save_array
    calls=[]
    def interrupted(path,value):
        calls.append(path.name)
        if len(calls)==3:
            raise OSError('Synthetic disk write failure')
        save(path,value)
    monkeypatch.setattr(segmentation,'_save_array',interrupted)
    photo=workflow.queue.photos[0]
    with pytest.raises(OSError,match='disk write failure'):
        segment(workflow,photo.id,parser=FixtureParser())
    root=workflow.session.work(photo.id,'segmentation')
    assert calls[:2]==['body-labels.npy','face-labels.npy']
    assert list(root.glob('.staging-*')) and not list(root.glob('*/manifest.json'))
    assert photo.render_attempts==[]


def test_approved_synthetic_semantic_recipe_exports_with_cached_masks(workflow):
    from conftest import import_review
    import tifffile
    cache,_=cached(workflow);path=semantic_plan(workflow,cache)
    photo=workflow.queue.photos[0]
    workflow.import_plan(photo.id,path);workflow.process(photo.id)
    import_review(workflow,photo)
    workflow.approve(photo.id,'v001','Synthetic fixture approval, never a real user photograph')
    before=digest(workflow.session.source(photo))
    workflow.export(photo.id)
    assert photo.state=='exported'
    assert digest(workflow.session.source(photo))==before
    with tifffile.TiffFile(workflow.session.path('final',photo.id+'.tif')) as handle:
        assert handle.pages[0].dtype==np.dtype('uint16')
    manifest=json.loads(workflow.session.path('final',photo.id+'.manifest.json').read_text())
    assert manifest['source_version']=='v001'

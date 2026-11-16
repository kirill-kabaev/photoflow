"""Offline synthetic texture/consent tests. No real acne classification is claimed."""
import json

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError
from scipy.ndimage import gaussian_filter

from conftest import make_plan
from test_director import complete_plan, directed_review
from photoflow import healing, imaging, queue
from photoflow.director import Director, DirectorConfig, review_context
from photoflow.models import NaturalSkinRetouch, Plan, Recipe, SpotHealing
from photoflow.segmentation import segment, SemanticCache
from photoflow.skin import natural_skin_retouch, preview_existing
from photoflow.storage import FlowError, digest, read_model, write_json


class FaceFixture:
    identity = {'id':'synthetic-face-labels-NOT-neural'}
    execution = {'adapter':'explicit-test-fixture'}
    timings = {}
    def predict(self, image):
        body = np.full((image.height,image.width),11,np.uint8)
        face = np.ones_like(body)
        return body,face,{'fixture':True}


@pytest.fixture
def skin_pixels(pixels):
    yy,xx = np.mgrid[:80,:120]
    texture = .003*np.sin(xx*1.8)*np.cos(yy*1.4)
    # Localized redness, and a separate permanent dark mark.
    spot = np.exp(-((xx-55)**2+(yy-40)**2)/18)
    mole = np.exp(-((xx-90)**2+(yy-55)**2)/3)
    for c,base in enumerate([.45,.30,.22]):
        pixels[...,c] = base+texture - .15*mole
    pixels[...,0] += .17*spot
    pixels[...,1] -= .04*spot
    return pixels


def fixture_version(workflow, skin_pixels, directed=False):
    photo = workflow.queue.photos[0]
    # decoder fixture closes over pixels; skin_pixels edits that same array.
    workflow.prepare(photo.id)
    result = segment(workflow,photo.id,parser=FaceFixture())
    path = complete_plan(workflow,photo) if directed else make_plan(workflow,photo,exposure=0.)
    data = json.loads(path.read_text())
    data['recipe'].update(segmentation={'cache_key':result['cache_key']},
        semantic_masks=[{'id':'skin','region':'face_skin','description':'Synthetic fixture, not neural'}],
        detail_regions=[{'id':'face','bounds':{},'description':'Native synthetic face'}])
    write_json(path,data)
    workflow.import_plan(photo.id,path)
    workflow.process(photo.id)
    return photo


def choose(workflow, photo, scan, answer='Yes'):
    if scan.get('status') == 'needs_visual_review':
        folder, proposal = healing.load_proposal(workflow,photo,scan['request_id'])
        reviewed = healing.review_candidates(workflow,photo.id,proposal.id,[s.id for s in proposal.spots],
            'Explicit synthetic visual-review fixture, NOT real AI',proposal.overlay_sha256)
        scan.update(reviewed)
    return healing.decide(workflow,photo.id,scan['request_id'],answer,'test-user',answer,scan['overlay_sha256'])


def test_natural_skin_luminance_texture_and_protected_regions(skin_pixels):
    before = skin_pixels.copy()
    mask = np.zeros(before.shape[:2],np.float32); mask[8:72,12:108] = 1
    after = natural_skin_retouch(before.copy(),mask,NaturalSkinRetouch())
    assert np.max(np.abs(imaging.luminance(after)-imaging.luminance(before))) < 1e-7
    assert np.array_equal(before[mask==0],after[mask==0])
    assert np.array_equal(before[53:58,88:93],after[53:58,88:93])  # mole protected
    assert np.max(np.abs(after-before)) > 1e-4
    # Fine luminance texture is retained, rather than replaced by a blurred patch.
    assert np.allclose(imaging.luminance(after)-gaussian_filter(imaging.luminance(after),2),
                       imaging.luminance(before)-gaussian_filter(imaging.luminance(before),2),atol=1e-7)
    for settings in [NaturalSkinRetouch(enabled=False),NaturalSkinRetouch(tone_evenness=0.,redness=0.)]:
        assert np.array_equal(natural_skin_retouch(before.copy(),mask,settings),before)


def test_natural_skin_chroma_unevenness_reduces_without_removing_pores():
    yy,xx=np.mgrid[:120,:160]
    pixels=np.full((120,160,3),.3,np.float32)
    pixels += (.003*np.sin(xx*2)*np.cos(yy*1.7))[...,None]
    field=.025*np.exp(-((xx-80)**2+(yy-60)**2)/150)
    pixels[...,0]+=field; pixels[...,1]-=field*.2126/.7152
    original=pixels.copy()
    result=natural_skin_retouch(pixels,np.ones((120,160),np.float32),NaturalSkinRetouch(redness=0.))
    assert result[60,80,0]-result[60,80,1] < original[60,80,0]-original[60,80,1]
    assert np.max(np.abs(imaging.luminance(result)-imaging.luminance(original))) < 1e-7


def test_natural_skin_schema_no_global_skin_filter():
    with pytest.raises(ValidationError): NaturalSkinRetouch(redness=.8)
    with pytest.raises(ValidationError): NaturalSkinRetouch(reshape=True)
    assert Recipe(photo_id='x',source_sha256='a'*64,style='x',rationale='x').natural_skin is None


def test_detection_is_suspected_red_spot_not_dark_mark_or_mask_edge(skin_pixels):
    rgb=imaging.encode_srgb(skin_pixels)
    spots=healing.detect(rgb,np.ones(rgb.shape[:2],np.float32))
    assert spots and any(abs(s['x']-55)<3 and abs(s['y']-40)<3 for s in spots)
    assert all(s['status']=='suspected' for s in spots)
    assert not any((s['x']-90)**2+(s['y']-55)**2<15**2 for s in spots)
    assert healing.detect(rgb,np.zeros(rgb.shape[:2],np.float32))==[]
    assert healing.detect(np.full_like(rgb,.5),np.ones(rgb.shape[:2],np.float32))==[]


def test_scan_existing_native_preview_no_decode_pass_or_mutation(workflow,skin_pixels,monkeypatch):
    photo=fixture_version(workflow,skin_pixels)
    before=workflow.session.path('queue.json').read_bytes()
    old=workflow.session.work(photo.id,'v001','manifest.json').read_bytes()
    monkeypatch.setattr(workflow,'decoder',lambda *_:pytest.fail('Detection must not decode RAW'))
    result=healing.scan(workflow,photo.id)
    assert result['status']=='needs_visual_review' and result['count']>=1 and result['confirmed_defects']==0
    assert result['question'] is None
    assert healing.QUESTION=='Potential pimples detected. Remove them? Yes / No'
    assert healing.scan(workflow,photo.id)['request_id']==result['request_id']
    assert workflow.session.path('queue.json').read_bytes()==before
    assert workflow.session.work(photo.id,'v001','manifest.json').read_bytes()==old
    preview=preview_existing(workflow,photo.id)
    with Image.open(preview['after']) as face: assert face.size==(120,80)
    assert workflow.session.path('queue.json').read_bytes()==before


def test_yes_creates_budgeted_immutable_raw_version_and_native_crops(workflow,skin_pixels):
    photo=fixture_version(workflow,skin_pixels)
    assert read_model(workflow.session.work(photo.id,'v001','recipe.json'),Recipe).natural_skin.enabled
    original={p.name:digest(p) for p in workflow.session.work(photo.id,'v001').iterdir() if p.is_file()}
    request=healing.scan(workflow,photo.id)
    choose(workflow,photo,request)
    result=healing.apply_approved(workflow,photo.id,request['request_id'])
    assert result['version']=='v002' and photo.active_attempt_count==2
    folder,recipe,manifest=workflow.validate_version(photo,'v002')
    assert recipe.spot_healing and manifest['spot_healing']
    operation=json.loads((folder/'healing/operation.json').read_text())
    assert operation['scale']==1.0 and operation['source_box']==[0,0,120,80]
    with Image.open(folder/'healing/before.jpg') as face: assert face.size==(120,80)
    assert {p.name:digest(p) for p in workflow.session.work(photo.id,'v001').iterdir() if p.is_file()}==original
    assert photo.approved_version is None and photo.state=='review'
    assert not list(workflow.session.path('final').iterdir())
    with pytest.raises(FlowError,match='approved'): workflow.export(photo.id)
    context=review_context(workflow,photo)
    assert 'v002/healing/before_after.jpg' in context['required_artifacts']


@pytest.mark.parametrize('answer',[None,'No'])
def test_no_or_unanswered_never_heals(workflow,skin_pixels,answer):
    photo=fixture_version(workflow,skin_pixels)
    request=healing.scan(workflow,photo.id)
    folder, proposal=healing.load_proposal(workflow,photo,request['request_id'])
    request.update(healing.review_candidates(workflow,photo.id,proposal.id,[s.id for s in proposal.spots],
        'Synthetic inspection only',proposal.overlay_sha256))
    if answer: choose(workflow,photo,request,answer)
    with pytest.raises((FlowError,FileNotFoundError),match='Missing|Yes|No such file'):
        healing.apply_approved(workflow,photo.id,request['request_id'])
    assert photo.active_attempt_count==1 and photo.versions==['v001'] and photo.state=='review'


def test_consent_bad_preview_stale_grade_and_other_photo_refused(workflow,skin_pixels):
    photo=fixture_version(workflow,skin_pixels)
    request=healing.scan(workflow,photo.id)
    _, proposal=healing.load_proposal(workflow,photo,request['request_id'])
    request.update(healing.review_candidates(workflow,photo.id,proposal.id,[s.id for s in proposal.spots],
        'Synthetic inspection',proposal.overlay_sha256))
    with pytest.raises(FlowError,match='shown overlay'):
        healing.decide(workflow,photo.id,request['request_id'],'Yes','test','Yes','0'*64)
    decision=choose(workflow,photo,request)
    _,recipe,_=workflow.validate_version(photo,'v001')
    recipe=recipe.model_copy(deep=True)
    recipe.parent_version='v001'
    recipe.spot_healing=SpotHealing(request_id=request['request_id'],decision_sha256=decision['decision_sha256'])
    recipe.global_adjustments.exposure_ev=.9
    with pytest.raises(FlowError,match='stale'): healing.authorization(workflow,photo,recipe)
    other=photo.model_copy(update={'id':'another-photo'})
    with pytest.raises(FlowError,match='Missing'): healing.authorization(workflow,other,recipe)
    with pytest.raises(FlowError,match='immutable'):
        healing.decide(workflow,photo.id,request['request_id'],'No','test-user','No',request['overlay_sha256'])


def test_heal_changes_only_tiny_disks_and_preserves_protected_feature(workflow,skin_pixels):
    photo=fixture_version(workflow,skin_pixels)
    request=healing.scan(workflow,photo.id)
    _,proposal=healing.load_proposal(workflow,photo,request['request_id'])
    before=skin_pixels.copy()
    result=healing.heal_pixels(before.copy(),proposal,np.ones(before.shape[:2],np.float32))
    yy,xx=np.mgrid[:80,:120]; affected=np.zeros((80,120),bool)
    for s in proposal.spots: affected|=(xx-s.x)**2+(yy-s.y)**2<s.radius**2
    assert np.array_equal(before[~affected],result[~affected])
    assert np.max(np.abs(result-before))>1e-4
    unsafe=proposal.model_copy(update={'spots':[proposal.spots[0].model_copy(update={'x':90,'y':55})]})
    pixels=before.copy()
    with pytest.raises(FlowError,match='unsafe target'): healing.heal_pixels(pixels,unsafe,np.ones((80,120),np.float32))
    assert np.array_equal(pixels,before)
    recipe=Recipe(photo_id=photo.id,source_sha256=photo.source_sha256,style='test',rationale='test',
        spot_healing=SpotHealing(request_id=request['request_id'],decision_sha256='0'*64))
    with pytest.raises(FlowError,match='consent'): imaging.process_pixels(before,recipe)


def test_ai_can_reject_uncertain_candidates_but_not_grant_permission(workflow,skin_pixels):
    photo=fixture_version(workflow,skin_pixels)
    request=healing.scan(workflow,photo.id)
    with pytest.raises(FlowError,match='existing candidate'):
        healing.review_candidates(workflow,photo.id,request['request_id'],['invented'],'uncertain',request['overlay_sha256'])
    checked=healing.review_candidates(workflow,photo.id,request['request_id'],[],'All uncertain; preserve marks',request['overlay_sha256'])
    assert checked['status']=='none_detected'
    assert checked['count']==0 and photo.active_attempt_count==1
    with pytest.raises(FlowError,match='No current suspected'):
        choose(workflow,photo,checked)


def test_director_pauses_per_photo_no_continues_and_yes_respects_budget(workflow,skin_pixels):
    director=Director(workflow)
    photo=workflow.queue.photos[0]
    director.start('Synthetic test requested workflow',photo.id,DirectorConfig())
    fixture_version(workflow,skin_pixels,directed=True)
    action=director.advance()
    assert action['action']=='spot_healing_visual_review'
    assert director.advance()['request_id']==action['request_id']
    assert photo.active_attempt_count==1
    choose(workflow,photo,action,'No')
    assert director.advance()['action']=='visual_review'
    assert photo.active_attempt_count==1
    directed_review(workflow,photo)
    assert director.advance()['action']=='present_chat'


def test_director_yes_native_review_and_no_budget_bypass(workflow,skin_pixels):
    director=Director(workflow)
    photo=workflow.queue.photos[0]
    director.start('Synthetic test',photo.id,DirectorConfig(max_full_passes=1))
    fixture_version(workflow,skin_pixels,directed=True)
    action=director.advance()
    choose(workflow,photo,action)
    stopped=director.advance()
    assert stopped['action']=='requires_user_input'
    assert 'No remaining' in stopped['reason']
    assert photo.active_attempt_count==1 and photo.current_version=='v001'


def test_director_explicit_yes_uses_second_slot_then_requires_native_review(workflow,skin_pixels):
    director=Director(workflow)
    photo=workflow.queue.photos[0]
    director.start('Synthetic test',photo.id)
    fixture_version(workflow,skin_pixels,directed=True)
    action=director.advance()
    choose(workflow,photo,action)
    review=director.advance()
    assert review['action']=='visual_review' and photo.current_version=='v002'
    assert 'v002/healing/before_after.jpg' in review['required_artifacts']
    assert photo.active_attempt_count==2 and photo.approved_version is None
    # A resumed Director must not repeat healing or reserve a third attempt.
    assert Director(workflow).advance()['action']=='visual_review'
    assert photo.active_attempt_count==2


def test_native_crop_coordinates_map_to_uncropped_raw(workflow,skin_pixels):
    photo=workflow.queue.photos[0]
    workflow.prepare(photo.id)
    result=segment(workflow,photo.id,parser=FaceFixture())
    path=make_plan(workflow,photo,exposure=0.)
    data=json.loads(path.read_text())
    data['recipe'].update(segmentation={'cache_key':result['cache_key']},
        semantic_masks=[{'id':'skin','region':'face_skin','description':'Synthetic'}],
        detail_regions=[{'id':'face','bounds':{'left':.2,'top':.2,'right':.8,'bottom':.8},'description':'Offset native ROI'}])
    write_json(path,data); workflow.import_plan(photo.id,path); workflow.process(photo.id)
    request=healing.scan(workflow,photo.id)
    _,proposal=healing.load_proposal(workflow,photo,request['request_id'])
    assert proposal.source_box==[24,16,96,64]
    assert any(abs(s.x-55)<=2 and abs(s.y-40)<=2 for s in proposal.spots)
    choose(workflow,photo,request)
    healing.apply_approved(workflow,photo.id,request['request_id'])
    with Image.open(workflow.session.work(photo.id,'v002','healing','after.jpg')) as crop:
        assert crop.size==(72,48)


def test_five_photo_batch_decisions_are_independent(workflow,skin_pixels):
    # Five synthetic originals, never the user's other pending RAW photographs.
    for i in range(4):
        workflow.session.path('raw',f'frame{i}.ARW').write_bytes(f'synthetic fixture {i}'.encode())
    workflow.queue=queue.scan(workflow.session)
    director=Director(workflow)
    director.start('User explicitly requested synthetic batch of five')
    ids=list(director.run.photo_ids)
    assert len(ids)==5
    for index, identifier in enumerate(ids):
        photo=workflow.photo(identifier)
        workflow.prepare(photo.id)
        result=segment(workflow,photo.id,parser=FaceFixture())
        path=complete_plan(workflow,photo)
        data=json.loads(path.read_text())
        data['recipe'].update(segmentation={'cache_key':result['cache_key']},
            semantic_masks=[{'id':'skin','region':'face_skin','description':'Synthetic fixture'}],
            detail_regions=[{'id':'face','bounds':{},'description':'Native synthetic face'}])
        write_json(path,data); workflow.import_plan(photo.id,path)
        action=director.advance()
        assert action['action']=='spot_healing_visual_review' and action['photo_id']==identifier
        assert photo.active_attempt_count==1  # Earlier Yes never authorizes this photo.
        choose(workflow,photo,action,'Yes' if index in {1,3} else 'No')
        assert director.advance()['action']=='visual_review'
        assert photo.active_attempt_count==(2 if index in {1,3} else 1)
        directed_review(workflow,photo)
        presented=director.advance()
        assert presented['action']=='present_chat'
        if index in {1,3}:
            assert 'healing/before_after.jpg' in presented['bundle']['artifacts']
        director.presented('tool_image_content')
        next_action=director.advance()
        assert next_action['action']==('complete' if index==4 else 'analyze_and_plan')
    assert all(p.approved_version is None for p in workflow.queue.photos)
    assert not list(workflow.session.path('final').iterdir())


def test_healing_approved_export_replays_raw_without_new_artistic_attempt(workflow,skin_pixels):
    from conftest import import_review
    photo=fixture_version(workflow,skin_pixels)
    request=healing.scan(workflow,photo.id); choose(workflow,photo,request)
    healing.apply_approved(workflow,photo.id,request['request_id'])
    import_review(workflow,photo)
    workflow.approve(photo.id,'v002','synthetic-fixture-user')
    exported=workflow.export(photo.id)
    assert exported['source_version']=='v002' and photo.active_attempt_count==2


def test_review_rejects_superseded_consent_and_artifact_tampering(workflow,skin_pixels):
    photo=fixture_version(workflow,skin_pixels)
    result=healing.scan(workflow,photo.id)
    folder,proposal=healing.load_proposal(workflow,photo,result['request_id'])
    reviewed=healing.review_candidates(workflow,photo.id,proposal.id,[proposal.spots[0].id],
                                       'Synthetic inspection retains one',proposal.overlay_sha256)
    with pytest.raises(FlowError,match='superseded|visual review'):
        healing.decide(workflow,photo.id,result['request_id'],'Yes','test-user','Yes',result['overlay_sha256'])
    target,_=healing.load_proposal(workflow,photo,reviewed['request_id'])
    (target/'suspected.jpg').write_bytes(b'corrupt overlay')
    with pytest.raises(FlowError,match='visual input'):
        choose(workflow,photo,reviewed)


@pytest.mark.parametrize('feature',[2,3,4,5,6,7,8,9,10,11,12,13,17])
def test_strict_sensitive_feature_guard_rejects_even_strong_red_spot(feature):
    from types import SimpleNamespace
    h,w=160,200
    face=np.ones((h,w),np.uint8)
    face[70:85,90:110]=feature
    cache=SimpleNamespace(face=face,mask=lambda name,shape:np.ones(shape[:2],np.float32))
    safe=healing.eligible_skin(cache,(h,w,3))
    assert np.all(safe[58:97,78:122]==0)
    yy,xx=np.mgrid[:h,:w]
    red=np.exp(-((xx-95)**2+(yy-77)**2)/18)
    rgb=np.full((h,w,3),.55,np.float32)
    rgb[...,0]+=.13*red
    rgb[...,1]-=.04*red
    assert healing.detect(rgb,safe)==[]


def test_false_positive_regression_normal_pores_freckles_shadows_and_fold():
    rng=np.random.default_rng(17)
    h,w=160,200
    yy,xx=np.mgrid[:h,:w]
    rgb=np.full((h,w,3),.6,np.float32)
    rgb += rng.normal(0,.004,(h,w,1)).astype(np.float32)  # pores/noise
    diffuse=.015*np.exp(-((xx-80)**2+(yy-70)**2)/1200)
    rgb[...,0]+=diffuse  # natural broad colour variation
    for x,y in [(45,50),(75,100),(140,60)]:
        mark=np.exp(-((xx-x)**2+(yy-y)**2)/4)
        rgb-=mark[...,None]*np.array([.08,.10,.10],np.float32)
    shadow=.045*np.exp(-((xx-120)**2)/50)
    rgb-=shadow[...,None]  # elongated shadow/fold
    assert healing.detect(rgb,np.ones((h,w),np.float32))==[]


def test_unreviewed_machine_candidates_cannot_be_proposed_or_consented(workflow,skin_pixels):
    photo=fixture_version(workflow,skin_pixels)
    request=healing.scan(workflow,photo.id)
    assert request['status']=='needs_visual_review' and request['question'] is None
    with pytest.raises(FlowError,match='visual review'):
        healing.decide(workflow,photo.id,request['request_id'],'Yes','test-user','Yes',request['overlay_sha256'])
    assert photo.active_attempt_count==1


def test_natural_skin_healing_and_approved_crop_share_one_second_raw_pass(workflow,skin_pixels):
    from photoflow import cropping
    from test_cropping import analysis,approve
    photo=fixture_version(workflow,skin_pixels)
    spot=healing.scan(workflow,photo.id); consent=choose(workflow,photo,spot)
    proposal=cropping.propose(workflow,analysis(workflow,photo,'v001'))
    selected=approve(workflow,photo,proposal)
    folder,recipe,_=workflow.validate_version(photo,'v001')
    request=workflow.session.work(photo.id,'combined-user-request.md')
    request.write_text('Synthetic user approved only displayed spot and proposed crop')
    workflow.request_revision(photo.id,feedback_file=request)
    recipe=recipe.model_copy(deep=True)
    recipe.parent_version='v001'
    recipe.spot_healing=SpotHealing(request_id=spot['request_id'],decision_sha256=consent['decision_sha256'])
    recipe.feedback_applied='Synthetic explicit crop and healing consent; grade retained'
    recipe.feedback_sha256=digest(workflow.feedback_path(photo))
    plan=Plan.model_validate_json(json.dumps({'analysis':json.loads((folder/'analysis.json').read_text()),'recipe':recipe.model_dump(mode='json')}))
    plan=cropping.apply_selection(plan,selected)
    path=workflow.session.work(photo.id,'combined-plan.json'); write_json(path,plan)
    workflow.import_plan(photo.id,path); workflow.process(photo.id)
    _,recipe,_=workflow.validate_version(photo,'v002')
    assert recipe.natural_skin.enabled and recipe.spot_healing and recipe.approved_crop
    assert photo.active_attempt_count==2 and photo.approved_version is None

"""Offline action-protocol fixtures. These tests never claim real visual AI quality."""
import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from conftest import make_plan
from photoflow import queue
from photoflow.cache import RawCache
from photoflow.chat_reporting import verify_chat_report
from photoflow.cli import main
from photoflow.director import (ANALYSIS_CATEGORIES, REVIEW_CATEGORIES, Director,
                               DirectorConfig, review_context)
from photoflow.models import Plan, RawSettings
from photoflow.storage import FlowError, digest, write_json
from photoflow.workflow import Workflow


def complete_plan(workflow, photo, exposure=0.25):
    parent = workflow.revision_parent(photo)
    path = make_plan(workflow, photo, parent=parent, exposure=exposure,
                     feedback=workflow.feedback_path(photo) if parent else None)
    data = json.loads(path.read_text())
    data['analysis']['coverage'] = {key: 'Synthetic test-only observation, not real visual analysis' for key in ANALYSIS_CATEGORIES}
    data['recipe']['style'] = 'Natural Commercial Portrait'
    data['recipe']['detail_regions'] = [{'id': 'detail', 'bounds': {'left': 0.2, 'top': 0.2, 'right': 0.8, 'bottom': 0.8},
                                        'description': 'Synthetic test ROI'}]
    write_json(path, data)
    return path


def directed_review(workflow, photo, recommendation='human_review', best=None, intervention='none'):
    context = review_context(workflow, photo)
    best = best or photo.current_version
    inspected = dict(context['required_artifacts'])
    inspected[f'{best}/after.jpg'] = digest(workflow.session.work(photo.id, best, 'after.jpg'))
    data = {'photo_id': photo.id, 'version': photo.current_version,
        'comparison_sha256': digest(workflow.session.work(photo.id, photo.current_version, 'before_after.jpg')),
        'provider': 'test_fixture', 'model': 'synthetic-no-vision-model', 'visually_inspected': True,
        'improvements': ['Synthetic review fixture'], 'concerns': [], 'recommendation': recommendation,
        'rationale': 'Test of orchestration, not real visual quality',
        'assessment': {key: 'Synthetic fixture assessment' for key in REVIEW_CATEGORIES},
        'inspected_artifacts': inspected, 'full_resolution_texture_inspected': True,
        'best_version': best, 'selection_rationale': 'Explicit fixture comparison; newest is not automatically best',
        'intervention': intervention}
    if recommendation == 'revise':
        data['corrections'] = [{'problem': 'Synthetic dark face', 'region': 'Test region',
          'proposed_correction': 'Raise exposure slightly', 'expected_improvement': 'More readable midtones',
          'side_effects': ['Highlight risk'], 'verification': 'Inspect comparison and detail crop',
          'adjustment_paths': ['global_adjustments.exposure_ev']}]
    path = workflow.session.work(photo.id, 'test-directed-review.json')
    write_json(path, data)
    workflow.review(photo.id, path)
    return path


def start(workflow, config=None):
    director = Director(workflow)
    result = director.start('High-level user request: professionally process this photo',
                            workflow.queue.photos[0].id, config)
    assert result['action'] == 'analyze_and_plan'
    return director, workflow.queue.photos[0]


def render(workflow, director, photo, exposure=0.25):
    workflow.import_plan(photo.id, complete_plan(workflow, photo, exposure))
    result = director.advance()
    assert result['action'] == 'visual_review'
    return result


def test_one_successful_pass_stops_and_presents_actual_selected_images(workflow):
    director, photo = start(workflow)
    render(workflow, director, photo)
    directed_review(workflow, photo)
    action = director.advance()
    assert action['action'] == 'present_chat'
    bundle = action['bundle']
    assert bundle['version'] == 'v001' and bundle['approval_required'] is True
    assert bundle['frontend_rendering_verified'] is False
    assert Path(bundle['artifacts']['before.jpg']).is_file()
    assert Path(bundle['artifacts']['after.jpg']).is_file()
    verify_chat_report(workflow, bundle)
    director.presented('tool_image_content')
    assert director.advance()['action'] == 'complete'
    assert len(photo.render_attempts) == 1 and photo.render_attempts[0].status == 'complete'
    assert photo.state == 'review' and photo.approved_version is None
    assert not list(workflow.session.path('final').iterdir())


def test_optional_targeted_second_pass_can_select_older_version(workflow):
    director, photo = start(workflow)
    render(workflow, director, photo)
    first_recipe = workflow.session.work(photo.id, 'v001', 'recipe.json').read_bytes()
    directed_review(workflow, photo, recommendation='revise')
    assert director.advance()['action'] == 'targeted_correction_plan'
    render(workflow, director, photo, exposure=0.35)
    directed_review(workflow, photo, best='v001')
    action = director.advance()
    assert action['bundle']['version'] == 'v001'
    assert photo.current_version == 'v002' and photo.selected_version == 'v001'
    assert len(photo.render_attempts) == 2
    assert workflow.session.work(photo.id, 'v001', 'recipe.json').read_bytes() == first_recipe
    assert 'AI выбрал v001' in workflow.session.path('review.md').read_text()


def test_second_unsatisfactory_pass_stops_and_keeps_best(workflow):
    director, photo = start(workflow)
    render(workflow, director, photo)
    directed_review(workflow, photo, recommendation='revise')
    director.advance()
    render(workflow, director, photo, exposure=0.35)
    directed_review(workflow, photo, recommendation='revise', best='v001')
    stopped = director.advance()
    assert stopped['action'] == 'requires_user_input'
    assert photo.selected_version == 'v001' and len(photo.render_attempts) == 2
    workflow.request_revision(photo.id)
    workflow.import_plan(photo.id, complete_plan(workflow, photo, exposure=0.4))
    with pytest.raises(FlowError, match='pass limit'):
        workflow.process(photo.id)
    assert not workflow.session.work(photo.id, 'v003').exists()


def test_one_pass_configuration_stops_before_any_correction(workflow):
    director, photo = start(workflow, DirectorConfig(max_full_passes=1))
    render(workflow, director, photo)
    directed_review(workflow, photo, recommendation='revise')
    assert director.advance()['action'] == 'requires_user_input'
    assert len(photo.render_attempts) == 1


def test_incomplete_initial_analysis_and_unknown_recipe_are_rejected(workflow):
    director, photo = start(workflow)
    path = make_plan(workflow, photo)
    with pytest.raises(FlowError, match='style|incomplete'):
        workflow.import_plan(photo.id, path)
    path = complete_plan(workflow, photo)
    data = json.loads(path.read_text())
    data['recipe']['execute_python'] = 'malicious()'
    write_json(path, data)
    with pytest.raises(ValidationError):
        workflow.import_plan(photo.id, path)
    assert len(photo.render_attempts) == 0


def test_correction_cannot_silently_change_unrecommended_parameters(workflow):
    director, photo = start(workflow)
    render(workflow, director, photo)
    directed_review(workflow, photo, recommendation='revise')
    director.advance()
    path = complete_plan(workflow, photo, exposure=0.35)
    data = json.loads(path.read_text())
    data['recipe']['global_adjustments']['saturation'] = 1.3
    write_json(path, data)
    with pytest.raises(FlowError, match='only recommended'):
        workflow.import_plan(photo.id, path)
    assert len(photo.render_attempts) == 1


def test_preview_reuses_cache_and_never_consumes_full_passes(workflow):
    director, photo = start(workflow)
    path = complete_plan(workflow, photo)
    first = director.explore(photo.id, path, 'candidate-a')
    second = director.explore(photo.id, path, 'candidate-b')
    assert all(e['hit'] for e in first['raw_cache'] + second['raw_cache'])
    assert first['resolution'] == 'preview_only'
    assert first['linear_preview_dimensions'] == [60, 40]
    assert len(photo.render_attempts) == 0 and photo.state == 'analyzing'
    assert Path(first['preview']).is_file()
    director.explore(photo.id, path, 'candidate-c')
    with pytest.raises(FlowError, match='budget exhausted'):
        director.explore(photo.id, path, 'candidate-d')
    assert not workflow.session.work(photo.id, 'v001').exists()


def test_resume_does_not_reset_pass_budget_or_duplicate_render(workflow):
    director, photo = start(workflow)
    render(workflow, director, photo)
    reopened = Workflow(workflow.session, workflow.decoder)
    resumed = Director(reopened)
    assert resumed.advance()['action'] == 'visual_review'
    assert len(reopened.photo(photo.id).render_attempts) == 1
    with pytest.raises(FlowError, match='resume'):
        resumed.start('try to bypass budget', photo.id)


def test_failure_is_recorded_and_not_silently_skipped(workflow):
    director, photo = start(workflow)
    workflow.import_plan(photo.id, complete_plan(workflow, photo))
    def broken_decoder(*args):
        raise FlowError('Simulated decoder failure')
    workflow.decoder = broken_decoder
    assert director.advance()['action'] == 'requires_user_input'
    persisted = queue.load(workflow.session).photos[0]
    assert persisted.render_attempts[0].status == 'failed'
    assert director.advance()['action'] == 'requires_user_input'
    assert len(persisted.render_attempts) == 1


def test_batch_continues_after_presentation_without_approval(workflow):
    workflow.session.path('raw', 'other.ARW').write_bytes(b'another synthetic original')
    queue.scan(workflow.session)
    workflow = Workflow(workflow.session, workflow.decoder)
    director = Director(workflow)
    action = director.start('Process remaining synthetic shoot')
    photo = workflow.photo(action['photo_id'])
    render(workflow, director, photo)
    directed_review(workflow, photo)
    assert director.advance()['action'] == 'present_chat'
    director.presented('artifact_links')
    other = next(p for p in workflow.queue.photos if p.id != photo.id)
    assert other.state == 'pending'  # Acknowledgement itself does not process another photo.
    next_action = director.advance()
    assert next_action['photo_id'] == other.id and next_action['action'] == 'analyze_and_plan'
    assert photo.state == 'review' and photo.approved_version is None


@pytest.mark.parametrize('intervention', ['artistic_ambiguity', 'identity_risk', 'missing_capability'])
def test_exception_based_intervention(workflow, intervention):
    director, photo = start(workflow)
    render(workflow, director, photo)
    directed_review(workflow, photo, intervention=intervention)
    assert director.advance()['action'] == 'requires_user_input'
    assert len(photo.render_attempts) == 1


def test_stale_report_and_false_texture_claim_are_rejected(workflow):
    director, photo = start(workflow)
    render(workflow, director, photo)
    path = directed_review(workflow, photo)
    data = json.loads(path.read_text())
    data['inspected_artifacts'].pop(f'{photo.current_version}/details/detail/before_after.jpg')
    write_json(path, data)
    with pytest.raises(FlowError, match='actual inspected artifact'):
        workflow.review(photo.id, path)
    directed_review(workflow, photo)
    bundle = director.advance()['bundle']
    data = dict(bundle); data['artifacts'] = dict(bundle['artifacts']); data['artifacts']['after.jpg'] = '/tmp/unrelated.jpg'
    with pytest.raises(FlowError, match='invalid or changed'):
        verify_chat_report(workflow, data)


def test_source_and_cache_integrity_checked_on_cache_hits(workflow):
    photo = workflow.queue.photos[0]
    cache = RawCache(workflow.session, workflow.decoder)
    decoded = cache.get(photo, RawSettings())
    assert not decoded.pixels.flags.writeable
    path = workflow.session.work(photo.id, 'raw_cache', cache.events[0]['key'], 'pixels.npy')
    path.write_bytes(b'corrupted cache')
    with pytest.raises(FlowError, match='cache integrity'):
        cache.get(photo, RawSettings())


def test_changed_raw_settings_create_separate_cache_entries(workflow):
    photo = workflow.queue.photos[0]
    cache = RawCache(workflow.session, workflow.decoder)
    camera = cache.get(photo, RawSettings())
    cache.get(photo, RawSettings(white_balance='daylight'))
    cache.get(photo, RawSettings())
    assert [event['hit'] for event in cache.events] == [False, False, True]
    assert cache.events[0]['key'] != cache.events[1]['key']


def test_pass_limit_cannot_be_configured_above_two():
    with pytest.raises(ValidationError):
        DirectorConfig(max_full_passes=3)


def test_api_mode_fails_honestly_and_does_not_upload(workflow, capsys):
    assert main(['direct',str(workflow.session.root),'--photo',workflow.queue.photos[0].id,'--mode','api']) == 1
    assert 'No image upload occurred' in capsys.readouterr().err


def test_interrupted_publish_recovery_keeps_reserved_pass_and_resumes_review(workflow):
    director, photo = start(workflow)
    render(workflow, director, photo)
    q = queue.load(workflow.session)
    record = q.photos[0]
    record.state, record.current_version, record.versions, record.operation_version = 'processing', None, [], 'v001'
    record.render_attempts[0].status = 'reserved'
    queue.save(workflow.session, q)
    reopened = Workflow(workflow.session, workflow.decoder)
    resumed = Director(reopened)
    assert resumed.advance()['action'] == 'recover'
    assert reopened.recover()[0]['state'] == 'review'
    assert resumed.advance()['action'] == 'visual_review'
    assert len(reopened.queue.photos[0].render_attempts) == 1
    assert reopened.queue.photos[0].render_attempts[0].status == 'complete'


def test_failed_first_batch_photo_blocks_continuation(workflow):
    workflow.session.path('raw', 'other.ARW').write_bytes(b'different synthetic source')
    q = queue.scan(workflow.session)
    first = q.photos[0]
    queue.transition(first, 'failed', 'Existing decode failure')
    first.error = 'Existing decode failure'
    queue.save(workflow.session, q)
    reopened = Workflow(workflow.session, workflow.decoder)
    director = Director(reopened)
    action = director.start('Batch request')
    assert action['action'] == 'requires_user_input' and action['photo_id'] == first.id
    assert reopened.queue.photos[1].state == 'pending'
    assert director.run.completed == []


def test_chat_report_tampering_blocks_presentation_acknowledgement(workflow):
    director, photo = start(workflow)
    render(workflow, director, photo)
    directed_review(workflow, photo)
    bundle = director.advance()['bundle']
    path = Path(bundle['bundle_path'])
    data = json.loads(path.read_text())
    data['checksums']['after.jpg'] = '0' * 64
    write_json(path, data)
    with pytest.raises(FlowError, match='invalid or changed'):
        director.presented('artifact_links')
    assert director.run.completed == [] and photo.approved_version is None


def test_intervention_report_preserves_best_and_resolution_cannot_reset_passes(workflow):
    director, photo = start(workflow, DirectorConfig(max_full_passes=1))
    render(workflow, director, photo)
    directed_review(workflow, photo, recommendation='revise')
    stopped = director.advance()
    assert stopped['action'] == 'requires_user_input'
    assert stopped['bundle']['version'] == 'v001'
    assert stopped['bundle']['status'] == 'requires_user_input'
    assert director.resolve('User acknowledges remaining limitation')['action'] == 'requires_user_input'
    assert len(photo.render_attempts) == 1
    assert any(event['event'] == 'user_resolution' for event in director.run.events)


def test_preview_and_full_renderer_share_color_semantics(workflow, pixels):
    from photoflow import imaging
    from PIL import Image
    director, photo = start(workflow)
    path = complete_plan(workflow, photo)
    preview = director.explore(photo.id, path, 'color-semantics')
    plan = Plan.model_validate_json(path.read_bytes())
    full = imaging.process_pixels(pixels, plan.recipe)
    reference = np.asarray(imaging.to_preview(full, max_edge=60)).astype(float)
    with Image.open(Path(preview['preview']).parent/'after.jpg') as image:
        actual = np.asarray(image).astype(float)
    assert actual.shape == reference.shape
    assert np.mean(np.abs(actual-reference)) < 2.0  # JPEG quantization, not a different gamma/color pipeline.
    assert len(photo.render_attempts) == 0


def test_analysis_exploration_and_full_pass_decode_each_raw_setting_once(workflow):
    calls=[]
    original=workflow.decoder
    def counting_decoder(path, settings):
        calls.append(settings.model_dump())
        return original(path, settings)
    workflow.decoder=counting_decoder
    director, photo=start(workflow)
    path=complete_plan(workflow,photo)
    data=json.loads(path.read_text())
    data['recipe']['raw']={'white_balance':'daylight'}
    write_json(path,data)
    director.explore(photo.id,path,'wb-first')
    director.explore(photo.id,path,'wb-second')
    workflow.import_plan(photo.id,path)
    assert director.advance()['action']=='visual_review'
    assert len(calls)==2
    performance=json.loads(workflow.session.work(photo.id,'v001','metrics.json').read_text())['performance']
    assert all(event['hit'] for event in performance['raw_cache'])
    assert len(photo.render_attempts)==1


def test_legacy_ready_plan_is_reanalyzed_before_first_directed_pass(workflow):
    photo=workflow.queue.photos[0]
    workflow.prepare(photo.id)
    workflow.import_plan(photo.id,make_plan(workflow,photo))
    assert photo.state=='ready'
    director=Director(workflow)
    action=director.start('Complete commercial processing',photo.id)
    assert action['action']=='analyze_and_plan'
    assert photo.state=='analyzing' and photo.render_attempts==[]


def test_complete_analysis_is_revalidated_at_execution_not_only_import(workflow):
    director,photo=start(workflow)
    workflow.import_plan(photo.id,complete_plan(workflow,photo))
    path=workflow.session.work(photo.id,'plan.json')
    data=json.loads(path.read_text()); data['analysis']['coverage']={}
    write_json(path,data)
    with pytest.raises(FlowError,match='incomplete'):
        workflow.process(photo.id)
    assert photo.render_attempts==[]


def test_review_selection_recovers_if_review_was_saved_before_queue_metadata(workflow):
    director,photo=start(workflow)
    render(workflow,director,photo)
    directed_review(workflow,photo)
    q=queue.load(workflow.session)
    q.photos[0].selected_version=None
    q.photos[0].selection_rationale=''
    queue.save(workflow.session,q)
    reopened=Workflow(workflow.session,workflow.decoder)
    resumed=Director(reopened)
    assert resumed.advance()['action']=='present_chat'
    assert reopened.queue.photos[0].selected_version=='v001'
    assert len(reopened.queue.photos[0].render_attempts)==1


def test_direct_next_scopes_one_photo_without_silently_skipping_failure(workflow, capsys):
    workflow.session.path('raw','other.ARW').write_bytes(b'another synthetic RAW')
    q=queue.scan(workflow.session)
    first=q.photos[0]
    queue.transition(first,'failed','Existing failure must not be skipped')
    first.error='Existing failure must not be skipped'
    queue.save(workflow.session,q)
    assert main(['direct',str(workflow.session.root),'--next'])==0
    action=json.loads(capsys.readouterr().out)
    assert action['action']=='requires_user_input' and action['photo_id']==first.id
    run=Director(Workflow(workflow.session,workflow.decoder)).run
    assert run.photo_ids==[first.id]
    assert queue.load(workflow.session).photos[1].state=='pending'

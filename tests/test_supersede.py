import uuid
import pytest
from conftest import process_first
from photoflow.director import Director,DirectorRun,DirectorConfig
from photoflow.storage import FlowError,now


def active(workflow,photo):
    director=Director(workflow)
    director.run=DirectorRun(id=uuid.uuid4().hex,instruction='Synthetic earlier task',config=DirectorConfig(),
        photo_ids=[photo.id],active=photo.id,created_at=now(),updated_at=now())
    director.save()
    return director


def test_new_user_cycle_supersedes_run_without_fake_review_or_budget_reset(workflow):
    p=process_first(workflow);director=active(workflow,p)
    snapshot=p.model_dump()
    with pytest.raises(FlowError,match='existing Director'):
        workflow.begin_cycle(p.id,'user','New explicit work',1)
    director.supersede('user','Explicit new global brightness phase')
    assert p.model_dump()==snapshot
    assert director.advance()['action']=='superseded'
    workflow.begin_cycle(p.id,'user','Explicit new global brightness phase',1)
    assert len(p.render_attempts)==1 and p.active_attempt_count==0 and p.active_pass_limit==1
    assert not p.approved_version
    with pytest.raises(FlowError,match='completed/failed'):
        workflow.begin_cycle(p.id,'user','No attempts in current cycle yet',1)


def test_supersede_refuses_missing_provenance_or_inflight_operation(workflow):
    p=process_first(workflow);director=active(workflow,p)
    with pytest.raises(FlowError,match='provenance'):director.supersede('','new task')
    p.state='processing'
    with pytest.raises(FlowError,match='in-flight'):director.supersede('user','new task')
    assert director.run.status=='active'

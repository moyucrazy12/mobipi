"""Tasks the RB-Y1 pipeline can run, by --task name. Add new PipelineTask subclasses here."""

from rby1.pipeline.tasks.base import PipelineTask
from rby1.pipeline.tasks.close_drawer import CloseDrawer
from rby1.pipeline.tasks.close_microwave import CloseMicrowave

TASKS = {task.name: task for task in (CloseMicrowave(), CloseDrawer())}


def get_task(name):
    return TASKS[name]

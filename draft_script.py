from audioop import reverse
import os
import inspect
from matplotlib import type1font

from numba.experimental import jitclass
from skglm.penalties.separable import L1
from skglm.datafits import QuadraticMultiTask


class TestClass:
    def __init__(self, user) -> None:
        self.user = user

    def greet(self) -> None:
        print('Hello world')


def _get_callable_member(obj):
    for attr in reversed(dir(obj)):
        if attr.startswith(('__', '_')):
            continue

        member = getattr(obj, attr)
        if callable(member):
            return member


package = 'skglm'
quad = QuadraticMultiTask()


member = _get_callable_member(QuadraticMultiTask)


fn = inspect.getsourcefile(member)
fn = os.path.relpath(fn, start=os.path.dirname(__import__(package).__file__))

lineno = inspect.getsourcelines(member)[1]

print(fn, lineno)

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""AbsoluteValue bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import bijector

__all__ = [
    "Gem",
]


class Gem(bijector.Bijector):

  def __init__(self, event_ndims=0, validate_args=False, name="gem"):
    self._graph_parents = []
    self._name = name

    event_ndims = ops.convert_to_tensor(event_ndims, name="event_ndims")
    event_ndims_const = tensor_util.constant_value(event_ndims)
    if event_ndims_const is not None and event_ndims_const not in (0,):
      raise ValueError("event_ndims(%s) was not 0" % event_ndims_const)
    else:
      if validate_args:
        event_ndims = control_flow_ops.with_dependencies(
            [check_ops.assert_equal(
                event_ndims, 0, message="event_ndims was not 0")],
            event_ndims)

    with self._name_scope("init"):
      super(Gem, self).__init__(
          event_ndims=event_ndims,
          validate_args=validate_args,
          name=name)

  def _forward(self, x):
    return math_ops.multiply(x, math_ops.cumprod(1. - x, exclusive=True, axis=-1))

  def _inverse(self, y):
    return math_ops.div(y, 1. - math_ops.cumsum(y, exclusive=True, axis=-1))

  def _forward_log_det_jacobian(self, x):
    return math_ops.log(\
             math_ops.abs(\
               math_ops.reduce_prod(\
                 math_ops.cumprod(1. - x, exclusive=True, axis=-1), axis=-1)))

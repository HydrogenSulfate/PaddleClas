# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddle import nn, Tensor
from typing import List
from .arcmargin import ArcMargin
from .cosmargin import CosMargin
from .circlemargin import CircleMargin
from .fc import FC
from .vehicle_neck import VehicleNeck
from paddle.nn import Tanh
from .bnneck import BNNeck
from .adamargin import AdaMargin

__all__ = ['build_gear']


class MultiLayer(nn.Layer):
    def __init__(self, layer_list: List[nn.Layer], output_name_list: List[str]):
        """MultiLayer

        Args:
            layer_list (List[nn.layer]): input layer list
        """
        super(MultiLayer, self).__init__()
        self.layer_list = nn.LayerList(layer_list)
        self.output_name_list = output_name_list

    def forward(self, input, label=None) -> List[Tensor]:
        output_dict = {}
        for i in range(len(self.layer_list)):
            output = self.layer_list[i](input) if label is None else self.layer_list[i](input, label)
            output_dict[self.output_name_list[i]] = output
        return output_dict


def build_gear(config):
    support_dict = [
        'ArcMargin', 'CosMargin', 'CircleMargin', 'FC', 'VehicleNeck', 'Tanh',
        'BNNeck', 'AdaMargin'
    ]
    if isinstance(config, dict):
        module_name = config.pop('name')
        assert module_name in support_dict, Exception(
            'head only support {}'.format(support_dict))
        module_class = eval(module_name)(**config)
        return module_class
    elif isinstance(config, list):
        module_class_list = []
        output_name_list = []
        for sub_config in config:
            module_name: str = list(sub_config.keys())[0]
            assert module_name in support_dict, Exception(
                'head only support {}'.format(support_dict))
            module_config: dict = sub_config[module_name]
            output_name = module_config.pop("output_name")

            output_name_list.append(output_name)
            module_class_list.append(eval(module_name)(**module_config))

        module_class_list = MultiLayer(module_class_list, output_name_list)
        return module_class_list
    else:
        raise TypeError(
            f"gear config must be dict or list, but got {type(config)}"
        )

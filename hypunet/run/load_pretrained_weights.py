#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import torch


def load_pretrained_weights(network, fname, verbose=False):
    """
    THIS DOES NOT TRANSFER SEGMENTATION HEADS!
    """
    saved_model = torch.load(fname)
    pretrained_dict = saved_model["state_dict"]
    # print(network)
    # print(list(pretrained_dict.keys()))

    new_state_dict = {}

    # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
    # match. Use heuristic to make it match
    for k, value in pretrained_dict.items():
        key = k
        print()
        print(key)
        if not key in list(network.state_dict().keys()):
            if "_orig_mod." in key:
                key = key.replace("_orig_mod.","")
            # remove module. prefix from DDP models
            if key.startswith("module."):
                key = key.replace("module.","")
        print(key)
        new_state_dict[key] = value

    pretrained_dict = new_state_dict

    model_dict = network.state_dict()
    ok = True
    for key, _ in model_dict.items():
        if "conv_blocks" in key:
            if (key in pretrained_dict) and (
                model_dict[key].shape == pretrained_dict[key].shape
            ):
                continue
            else:
                print()
                print(key)
                print(model_dict[key].shape)
                print(pretrained_dict[key].shape)
                ok = False
                break

    # filter unnecessary keys
    if ok:
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)
        }
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        print(
            "################### Loading pretrained weights from file ",
            fname,
            "###################",
        )
        if verbose:
            print(
                "Below is the list of overlapping blocks in pretrained model and hypunet architecture:"
            )
            for key, _ in pretrained_dict.items():
                print(key)
        print("################### Done ###################")
        network.load_state_dict(model_dict)
    else:
        raise RuntimeError(
            "Pretrained weights are not compatible with the current network architecture"
        )

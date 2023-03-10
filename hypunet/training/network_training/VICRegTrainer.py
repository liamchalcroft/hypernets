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


from itertools import chain

import torch
from hypunet.training.network_training.ContrastivePreTrainer import (
    ContrastivePreTrainer,
    GC_ContrastivePreTrainer,
)
from hypunet.training.network_training.custom_layer import BatchNormDimSwap
from batchgenerators.utilities.file_and_folder_operations import *

from solo.losses.vicreg import vicreg_loss_func
from grad_cache.functional import cat_input_tensor


class VICRegTrainer(ContrastivePreTrainer):
    """
    Info for Fabian: same as internal hypunetTrainerV2_2
    """

    def __init__(
        self,
        plans_file,
        output_folder=None,
        dataset_directory=None,
        unpack_data=True,
        deterministic=True,
        fp16=False,
        freeze_encoder=False,
        freeze_decoder=True,
        hyper_depth=None,
        meta_dim=2,
        proj_output_dim=2048,
        proj_hidden_dim=2048,
        sim_loss_weight=25.0,
        var_loss_weight=25.0,
        cov_loss_weight=1.0,
        detcon=False,
        batch_size=None,
    ):
        super().__init__(
            plans_file,
            output_folder,
            dataset_directory,
            unpack_data,
            deterministic,
            fp16,
            freeze_encoder,
            freeze_decoder,
            hyper_depth=hyper_depth,
            meta_dim=meta_dim,
            batch_size=batch_size,
        )

        self.load_plans_file()
        self.process_plans(self.plans)
        self.detcon = detcon

        # self.initial_lr = 1e-4

        self.projector = torch.nn.Sequential(
            torch.nn.Linear(320 if self.threeD else 480, proj_hidden_dim),
            BatchNormDimSwap(),
            torch.nn.BatchNorm1d(proj_hidden_dim),
            BatchNormDimSwap(),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_hidden_dim, proj_hidden_dim),
            BatchNormDimSwap(),
            torch.nn.BatchNorm1d(proj_hidden_dim),
            BatchNormDimSwap(),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        if torch.cuda.is_available():
            self.projector.cuda()

        self.sim_loss_weight = sim_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(
            chain(self.network.parameters(), self.projector.parameters()),
            self.initial_lr,
            weight_decay=self.weight_decay,
        )
        self.lr_scheduler = None

    def loss(self, view1, view2, mask1=None, mask2=None):
        if self.detcon:  # pool by multiplying images with masks
            view1, view2 = self.detcon_views(view1, view2, mask1, mask2)
        else:
            view1 = view1.view(view1.size(0), view1.size(1), -1).mean(dim=2)
            view2 = view2.view(view2.size(0), view2.size(1), -1).mean(dim=2)

        # print('before proj')
        # print('view1: ', view1.min(), view1.max(), view1.mean(), view1.std())
        # print('view2: ', view2.min(), view2.max(), view2.mean(), view2.std())

        z1 = self.projector(view1)
        z2 = self.projector(view2)

        if (
            self.detcon == "intra"
        ):  # treat each class as batch item - separate classes in same image will be treated as separate images
            z1 = z1.view(z1.size(0) * z1.size(1), -1)
            z2 = z2.view(z2.size(0) * z2.size(1), -1)
        elif (
            self.detcon == "inter"
        ):  # treat each class as batch and original batch as features - same class if diff images treated as same image
            z1 = z1.permute(1, 0, 2).reshape(z1.size(1), -1)
            z2 = z2.permute(1, 0, 2).reshape(z2.size(1), -1)

        # print('after proj')
        # print('z1: ', z1.min(), z1.max(), z1.mean(), z1.std())
        # print('z2: ', z2.min(), z2.max(), z2.mean(), z2.std())

        vic_loss = vicreg_loss_func(
            z1, z2, self.sim_loss_weight, self.var_loss_weight, self.cov_loss_weight
        )

        del z1, z2, view1, view2

        return vic_loss


class GC_VICRegTrainer(GC_ContrastivePreTrainer):
    """
    Info for Fabian: same as internal hypunetTrainerV2_2
    """

    def __init__(
        self,
        plans_file,
        output_folder=None,
        dataset_directory=None,
        unpack_data=True,
        deterministic=True,
        fp16=False,
        freeze_encoder=False,
        freeze_decoder=True,
        hyper_depth=None,
        meta_dim=2,
        proj_output_dim=2048,
        proj_hidden_dim=2048,
        sim_loss_weight=25.0,
        var_loss_weight=25.0,
        cov_loss_weight=1.0,
        metabatch=8,
        detcon=False,
        batch_size=None,
    ):
        super().__init__(
            plans_file,
            output_folder,
            dataset_directory,
            unpack_data,
            deterministic,
            fp16,
            freeze_encoder,
            freeze_decoder,
            hyper_depth,
            meta_dim,
            batch_size=batch_size,
        )

        self.load_plans_file()
        self.process_plans(self.plans)
        self.metabatch = int(metabatch)
        self.num_batches_per_epoch *= self.metabatch
        # self.initial_lr *= self.metabatch**0.5
        self.detcon = detcon

        self.projector = torch.nn.Sequential(
            torch.nn.Linear(320 if self.threeD else 480, proj_hidden_dim),
            BatchNormDimSwap(),
            torch.nn.BatchNorm1d(proj_hidden_dim),
            BatchNormDimSwap(),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_hidden_dim, proj_hidden_dim),
            BatchNormDimSwap(),
            torch.nn.BatchNorm1d(proj_hidden_dim),
            BatchNormDimSwap(),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        if torch.cuda.is_available():
            self.projector.cuda()

        self.sim_loss_weight = float(sim_loss_weight)
        self.var_loss_weight = float(var_loss_weight)
        self.cov_loss_weight = float(cov_loss_weight)

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(
            chain(self.network.parameters(), self.projector.parameters()),
            self.initial_lr,
            weight_decay=self.weight_decay,
        )
        self.lr_scheduler = None

    @cat_input_tensor
    def loss(self, view1, view2, mask1=None, mask2=None):
        if self.detcon:  # pool by multiplying images with masks
            view1, view2 = self.detcon_views(view1, view2, mask1, mask2)
        else:
            view1 = view1.view(view1.size(0), view1.size(1), -1).mean(dim=2)
            view2 = view2.view(view2.size(0), view2.size(1), -1).mean(dim=2)

        z1 = self.projector(view1)
        z2 = self.projector(view2)

        if (
            self.detcon == "intra"
        ):  # treat each class as batch item - separate classes in same image will be treated as separate images
            z1 = z1.view(z1.size(0) * z1.size(1), -1)
            z2 = z2.view(z2.size(0) * z2.size(1), -1)
        elif (
            self.detcon == "inter"
        ):  # treat each class as batch and original batch as features - same class if diff images treated as same image
            z1 = z1.permute(1, 0, 2).reshape(z1.size(1), -1)
            z2 = z2.permute(1, 0, 2).reshape(z2.size(1), -1)

        vic_loss = vicreg_loss_func(
            z1, z2, self.sim_loss_weight, self.var_loss_weight, self.cov_loss_weight
        )

        del z1, z2, view1, view2

        return vic_loss

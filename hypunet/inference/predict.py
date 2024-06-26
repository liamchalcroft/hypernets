import argparse
from copy import deepcopy
from typing import Tuple, Union, List

import numpy as np
from batchgenerators.augmentations.utils import resize_segmentation
from hypunet.inference.segmentation_export import (
    save_segmentation_nifti_from_softmax,
    save_segmentation_nifti,
)
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Process, Queue, Pool
import torch
import SimpleITK as sitk
import shutil
from hypunet.postprocessing.connected_components import (
    load_remove_save,
    load_postprocessing,
)
from hypunet.training.model_restore import load_model_and_checkpoint_files
from hypunet.training.network_training.HyperTrainer import HyperTrainer
from hypunet.utilities.one_hot_encoding import to_one_hot

def preprocess_save_to_queue(
    preprocess_fn,
    q,
    list_of_lists,
    output_files,
    segs_from_prev_stage,
    classes,
    transpose_forward,
):
    errors_in = []
    for i, l in enumerate(list_of_lists):
        output_file = output_files[i]
        print("preprocessing", output_file)
        try:
            print(l)
            d, m, s, dct = preprocess_fn(l)

            # Check if image is loaded correctly
            if d is None:
                print(f"Error: Failed to load data for {output_file}")
                errors_in.append(output_file)
                continue

            # Check if metadata is loaded correctly
            if dct is None:
                print(f"Error: Failed to load metadata for {output_file}")
                errors_in.append(output_file)
                continue
            
            print(f"Loaded data shape: {d.shape}, metadata: {dct}")

            if segs_from_prev_stage[i] is not None:
                assert isfile(segs_from_prev_stage[i]) and segs_from_prev_stage[i].endswith(".nii.gz"), (
                    "segs_from_prev_stage must point to a segmentation file"
                )
                seg_prev = sitk.GetArrayFromImage(sitk.ReadImage(segs_from_prev_stage[i]))
                img = sitk.GetArrayFromImage(sitk.ReadImage(l[0]))
                assert all([i == j for i, j in zip(seg_prev.shape, img.shape)]), (
                    "image and segmentation from previous stage don't have the same pixel array shape! "
                    "image: %s, seg_prev: %s" % (l[0], segs_from_prev_stage[i])
                )
                seg_prev = seg_prev.transpose(transpose_forward)
                seg_reshaped = resize_segmentation(seg_prev, d.shape[1:], order=1)
                seg_reshaped = to_one_hot(seg_reshaped, classes)
                d = np.vstack((d, seg_reshaped)).astype(np.float32)
            if np.prod(d.shape) > (2e9 / 4 * 0.85):
                print("This output is too large for python process-process communication. Saving output temporarily to disk")
                np.save(output_file[:-7] + ".npy", d)
                d = output_file[:-7] + ".npy"
            q.put((output_file, (d, dct)))
        except Exception as e:
            print(f"Exception occurred while processing {output_file}: {e}")
            errors_in.append(output_file)
    q.put("end")
    if len(errors_in) > 0:
        print("There were some errors in the following cases:", errors_in)
        print("These cases were ignored.")
    else:
        print("This worker has ended successfully, no errors to report")

def preprocess_multithreaded(
    trainer, list_of_lists, output_files, num_processes=2, segs_from_prev_stage=None
):
    if segs_from_prev_stage is None:
        segs_from_prev_stage = [None] * len(list_of_lists)

    num_processes = min(len(list_of_lists), num_processes)
    classes = list(range(1, trainer.num_classes))
    assert isinstance(trainer, HyperTrainer)
    q = Queue(1)
    processes = []
    for i in range(num_processes):
        pr = Process(
            target=preprocess_save_to_queue,
            args=(
                trainer.preprocess_patient,
                q,
                list_of_lists[i::num_processes],
                output_files[i::num_processes],
                segs_from_prev_stage[i::num_processes],
                classes,
                trainer.plans["transpose_forward"],
            ),
        )
        pr.start()
        processes.append(pr)

    try:
        end_ctr = 0
        while end_ctr != num_processes:
            item = q.get()
            if item == "end":
                end_ctr += 1
                continue
            else:
                yield item
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
            p.join()
        q.close()

def predict_cases(
    model,
    list_of_lists,
    output_filenames,
    folds,
    save_npz,
    num_threads_preprocessing,
    num_threads_nifti_save,
    segs_from_prev_stage=None,
    do_tta=True,
    mixed_precision=True,
    overwrite_existing=False,
    all_in_gpu=False,
    step_size=0.5,
    checkpoint_name="model_final_checkpoint",
    segmentation_export_kwargs: dict = None,
    disable_postprocessing: bool = False,
):
    print("emptying cuda cache")
    torch.cuda.empty_cache()

    print("loading parameters for folds,", folds)
    trainer, params = load_model_and_checkpoint_files(
        model, folds, mixed_precision=mixed_precision, checkpoint_name=checkpoint_name
    )

    print("starting preprocessing generator")
    preprocessing = preprocess_multithreaded(
        trainer,
        list_of_lists,
        output_filenames,
        num_threads_preprocessing,
        segs_from_prev_stage,
    )

    print("starting prediction...")
    for preprocessed in preprocessing:
        output_filename, (d, dct) = preprocessed
        
        # Check if metadata and image are loaded correctly
        if d is None or dct is None:
            print(f"Error: Failed to load data or metadata for {output_filename}")
            continue
        
        if isinstance(d, str):
            data = np.load(d)
            os.remove(d)
            d = data

        all_softmax_outputs = np.zeros(
            (len(params), trainer.num_classes, *d.shape[1:]), dtype=np.float16
        )
        all_seg_outputs = np.zeros((len(params), *d.shape[1:]), dtype=int)
        print("predicting", output_filename)

        for i, p in enumerate(params):
            trainer.load_checkpoint_ram(p, False)
            res = trainer.predict_preprocessed_data_return_seg_and_softmax(
                d,
                dct,
                do_mirroring=do_tta,
                mirror_axes=trainer.data_aug_params["mirror_axes"],
                use_sliding_window=True,
                step_size=step_size,
                use_gaussian=True,
                all_in_gpu=all_in_gpu,
                mixed_precision=mixed_precision,
            )
            if len(params) > 1:
                all_softmax_outputs[i] = res[1]
            all_seg_outputs[i] = res[0]

        if hasattr(trainer, "regions_class_order"):
            region_class_order = trainer.regions_class_order
        else:
            region_class_order = None
        assert region_class_order is None, (
            "predict_cases_fastest can only work with regular softmax predictions "
            "and is therefore unable to handle trainer classes with region_class_order"
        )

        print("aggregating predictions")
        if len(params) > 1:
            softmax_mean = np.mean(all_softmax_outputs, 0)
            seg = softmax_mean.argmax(0)
        else:
            seg = all_seg_outputs[0]

        print("applying transpose_backward")
        transpose_forward = trainer.plans.get("transpose_forward")
        if transpose_forward is not None:
            transpose_backward = trainer.plans.get("transpose_backward")
            seg = seg.transpose([i for i in transpose_backward])

        print("initializing segmentation export")
        results.append(
            pool.starmap_async(
                save_segmentation_nifti, ((seg, output_filename, dct, 0, None),)
            )
        )
        print("done")

    print("inference done. Now waiting for the segmentation export to finish...")
    _ = [i.get() for i in results]

    if not disable_postprocessing:
        results = []
        pp_file = join(model, "postprocessing.json")
        if isfile(pp_file):
            print("postprocessing...")
            shutil.copy(pp_file, os.path.dirname(output_filenames[0]))
            for_which_classes, min_valid_obj_size = load_postprocessing(pp_file)
            results.append(
                pool.starmap_async(
                    load_remove_save,
                    zip(
                        output_filenames,
                        output_filenames,
                        [for_which_classes] * len(output_filenames),
                        [min_valid_obj_size] * len(output_filenames),
                    ),
                )
            )
            _ = [i.get() for i in results]
        else:
            print(
                "WARNING! Cannot run postprocessing because the postprocessing file is missing. Make sure to run "
                "consolidate_folds in the output folder of the model first!\nThe folder you need to run this in is "
                "%s" % model
            )

    pool.close()
    pool.join()

def check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities):
    print("This model expects %d input modalities for each image" % expected_num_modalities)
    files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
    maybe_case_ids = np.unique([i[:-12] for i in files])
    remaining = deepcopy(files)
    missing = []

    for case_id in maybe_case_ids:
        expected_files = [f"{case_id}_{i:04d}.nii.gz" for i in range(expected_num_modalities)]
        for ef in expected_files:
            if ef not in remaining:
                missing.append(ef)
            else:
                remaining.remove(ef)

    if missing:
        print(f"Missing files: {missing}")
        raise ValueError(f"Not all expected files are present for case IDs: {maybe_case_ids}")

    return maybe_case_ids

def predict_from_folder(
    model: str,
    input_folder: str,
    output_folder: str,
    folds: Union[Tuple[int], List[int]],
    save_npz: bool,
    num_threads_preprocessing: int,
    num_threads_nifti_save: int,
    lowres_segmentations: Union[str, None],
    part_id: int,
    num_parts: int,
    tta: bool,
    mixed_precision: bool = True,
    overwrite_existing: bool = True,
    mode: str = "normal",
    overwrite_all_in_gpu: bool = None,
    step_size: float = 0.5,
    checkpoint_name: str = "model_final_checkpoint",
    segmentation_export_kwargs: dict = None,
    disable_postprocessing: bool = False,
):
    maybe_mkdir_p(output_folder)
    shutil.copy(join(model, "plans.pkl"), output_folder)

    assert isfile(join(model, "plans.pkl")), "Folder with saved model weights must contain a plans.pkl file"
    expected_num_modalities = load_pickle(join(model, "plans.pkl"))["num_modalities"]

    case_ids = check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities)
    output_files = [join(output_folder, i + ".nii.gz") for i in case_ids]
    all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
    list_of_lists = [
        [join(input_folder, i) for i in all_files if i[: len(j)].startswith(j) and len(i) == (len(j) + 12)]
        for j in case_ids
    ]

    if lowres_segmentations is not None:
        assert isdir(lowres_segmentations), "if lowres_segmentations is not None then it must point to a directory"
        lowres_segmentations = [join(lowres_segmentations, i + ".nii.gz") for i in case_ids]
        assert all([isfile(i) for i in lowres_segmentations]), (
            "not all lowres_segmentations files are present. (I was searching for case_id.nii.gz in that folder)"
        )
        lowres_segmentations = lowres_segmentations[part_id::num_parts]
    else:
        lowres_segmentations = None

    if mode == "normal":
        if overwrite_all_in_gpu is None:
            all_in_gpu = False
        else:
            all_in_gpu = overwrite_all_in_gpu

        return predict_cases(
            model,
            list_of_lists[part_id::num_parts],
            output_files[part_id::num_parts],
            folds,
            save_npz,
            num_threads_preprocessing,
            num_threads_nifti_save,
            lowres_segmentations,
            tta,
            mixed_precision=mixed_precision,
            overwrite_existing=overwrite_existing,
            all_in_gpu=all_in_gpu,
            step_size=step_size,
            checkpoint_name=checkpoint_name,
            segmentation_export_kwargs=segmentation_export_kwargs,
            disable_postprocessing=disable_postprocessing,
        )
    elif mode == "fast":
        if overwrite_all_in_gpu is None:
            all_in_gpu = False
        else:
            all_in_gpu = overwrite_all_in_gpu

        assert save_npz is False
        return predict_cases_fast(
            model,
            list_of_lists[part_id::num_parts],
            output_files[part_id::num_parts],
            folds,
            num_threads_preprocessing,
            num_threads_nifti_save,
            lowres_segmentations,
            tta,
            mixed_precision=mixed_precision,
            overwrite_existing=overwrite_existing,
            all_in_gpu=all_in_gpu,
            step_size=step_size,
            checkpoint_name=checkpoint_name,
            segmentation_export_kwargs=segmentation_export_kwargs,
            disable_postprocessing=disable_postprocessing,
        )
    elif mode == "fastest":
        if overwrite_all_in_gpu is None:
            all_in_gpu = False
        else:
            all_in_gpu = overwrite_all_in_gpu

        assert save_npz is False
        return predict_cases_fastest(
            model,
            list_of_lists[part_id::num_parts],
            output_files[part_id::num_parts],
            folds,
            num_threads_preprocessing,
            num_threads_nifti_save,
            lowres_segmentations,
            tta,
            mixed_precision=mixed_precision,
            overwrite_existing=overwrite_existing,
            all_in_gpu=all_in_gpu,
            step_size=step_size,
            checkpoint_name=checkpoint_name,
            disable_postprocessing=disable_postprocessing,
        )
    else:
        raise ValueError("unrecognized mode. Must be normal, fast or fastest")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        help="Must contain all modalities for each patient in the correct"
        " order (same as training). Files must be named "
        "CASENAME_XXXX.nii.gz where XXXX is the modality "
        "identifier (0000, 0001, etc)",
        required=True,
    )
    parser.add_argument(
        "-o", "--output_folder", required=True, help="folder for saving predictions"
    )
    parser.add_argument(
        "-m",
        "--model_output_folder",
        help="model output folder. Will automatically discover the folds "
        "that were run and use those as an ensemble",
        required=True,
    )
    parser.add_argument(
        "-f",
        "--folds",
        nargs="+",
        default="None",
        help="folds to use for prediction. Default is None "
        "which means that folds will be detected "
        "automatically in the model output folder",
    )
    parser.add_argument(
        "-z",
        "--save_npz",
        required=False,
        action="store_true",
        help="use this if you want to ensemble"
        " these predictions with those of"
        " other models. Softmax "
        "probabilities will be saved as "
        "compresed numpy arrays in "
        "output_folder and can be merged "
        "between output_folders with "
        "merge_predictions.py",
    )
    parser.add_argument(
        "-l",
        "--lowres_segmentations",
        required=False,
        default="None",
        help="if model is the highres "
        "stage of the cascade then you need to use -l to specify where the segmentations of the "
        "corresponding lowres unet are. Here they are required to do a prediction",
    )
    parser.add_argument(
        "--part_id",
        type=int,
        required=False,
        default=0,
        help="Used to parallelize the prediction of "
        "the folder over several GPUs. If you "
        "want to use n GPUs to predict this "
        "folder you need to run this command "
        "n times with --part_id=0, ... n-1 and "
        "--num_parts=n (each with a different "
        "GPU (for example via "
        "CUDA_VISIBLE_DEVICES=X)",
    )
    parser.add_argument(
        "--num_parts",
        type=int,
        required=False,
        default=1,
        help="Used to parallelize the prediction of "
        "the folder over several GPUs. If you "
        "want to use n GPUs to predict this "
        "folder you need to run this command "
        "n times with --part_id=0, ... n-1 and "
        "--num_parts=n (each with a different "
        "GPU (via "
        "CUDA_VISIBLE_DEVICES=X)",
    )
    parser.add_argument(
        "--num_threads_preprocessing",
        required=False,
        default=6,
        type=int,
        help="Determines many background processes will be used for data preprocessing. Reduce this if you "
        "run into out of memory (RAM) problems. Default: 6",
    )
    parser.add_argument(
        "--num_threads_nifti_save",
        required=False,
        default=2,
        type=int,
        help="Determines many background processes will be used for segmentation export. Reduce this if you "
        "run into out of memory (RAM) problems. Default: 2",
    )
    parser.add_argument(
        "--tta",
        required=False,
        type=int,
        default=1,
        help="Set to 0 to disable test time data "
        "augmentation (speedup of factor "
        "4(2D)/8(3D)), "
        "lower quality segmentations",
    )
    parser.add_argument(
        "--overwrite_existing",
        required=False,
        type=int,
        default=1,
        help="Set this to 0 if you need "
        "to resume a previous "
        "prediction. Default: 1 "
        "(=existing segmentations "
        "in output_folder will be "
        "overwritten)",
    )
    parser.add_argument("--mode", type=str, default="normal", required=False)
    parser.add_argument(
        "--all_in_gpu",
        type=str,
        default="None",
        required=False,
        help="can be None, False or True",
    )
    parser.add_argument(
        "--step_size", type=float, default=0.5, required=False, help="don't touch"
    )
    parser.add_argument(
        "--disable_mixed_precision",
        default=False,
        action="store_true",
        required=False,
        help="Predictions are done with mixed precision by default. This improves speed and reduces "
        "the required vram. If you want to disable mixed precision you can set this flag. Note "
        "that this is not recommended (mixed precision is ~2x faster!)",
    )

    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    part_id = args.part_id
    num_parts = args.num_parts
    model = args.model_output_folder
    folds = args.folds
    save_npz = args.save_npz
    lowres_segmentations = args.lowres_segmentations
    num_threads_preprocessing = args.num_threads_preprocessing
    num_threads_nifti_save = args.num_threads_nifti_save
    tta = args.tta
    step_size = args.step_size

    overwrite = args.overwrite_existing
    mode = args.mode
    all_in_gpu = args.all_in_gpu

    if lowres_segmentations == "None":
        lowres_segmentations = None

    if isinstance(folds, list):
        if folds[0] == "all" and len(folds) == 1:
            pass
        else:
            folds = [int(i) for i in folds]
    elif folds == "None":
        folds = None
    else:
        raise ValueError("Unexpected value for argument folds")

    if tta == 0:
        tta = False
    elif tta == 1:
        tta = True
    else:
        raise ValueError("Unexpected value for tta, Use 1 or 0")

    if overwrite == 0:
        overwrite = False
    elif overwrite == 1:
        overwrite = True
    else:
        raise ValueError("Unexpected value for overwrite, Use 1 or 0")

    assert all_in_gpu in ["None", "False", "True"]
    if all_in_gpu == "None":
        all_in_gpu = None
    elif all_in_gpu == "True":
        all_in_gpu = True
    elif all_in_gpu == "False":
        all_in_gpu = False

    predict_from_folder(
        model,
        input_folder,
        output_folder,
        folds,
        save_npz,
        num_threads_preprocessing,
        num_threads_nifti_save,
        lowres_segmentations,
        part_id,
        num_parts,
        tta,
        mixed_precision=not args.disable_mixed_precision,
        overwrite_existing=overwrite,
        mode=mode,
        overwrite_all_in_gpu=all_in_gpu,
        step_size=step_size,
    )

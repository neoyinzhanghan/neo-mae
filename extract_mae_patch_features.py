import os
from pathlib import Path
import torch
from tqdm import tqdm
import argparse
from time import time
import pandas as pd
from datetime import datetime
import numpy as np
import h5py
import ray
from ray import tune
import multiprocessing
import sys

from yatt.wsi_files import find_wsi_paths
from yatt.pipe.utils import find_fpaths_with_matching_folder
from yatt.pipe.patch_grid import load_patch_grid_coords, load_wsi_patch_metadata
from yatt.wsi.core_ray import get_several_patches_at_mpp_ray
from yatt.nn.extract_feats_ray import NormalizeImageNp, FeatureExtractor
from yatt.h5_utils import add_array_batch_to_h5
from yatt.patch import parse_patch_size
from yatt.nn.guess_largest_batch_size import guess_largest_batch_size
from models_mae import load_model

parser = argparse.ArgumentParser(
    description="Extracts patch features from a WSI. Assumes you have already created the patch grid."
)

#########################################################
group = parser.add_argument_group("Path setup")
#########################################################

group.add_argument("--wsi_dir", type=str, help="Directory where the WSI files live.")

group.add_argument(
    "--patch_dir",
    type=str,
    help="Directory containing the WSI patch folders. Assumes each WSI folder is called after the WSI file name.",
)

group.add_argument(
    "--save_dir",
    type=str,
    help="Directory where the patch features will be saved. One HDF5 file will be saved per WSI.",
)

parser.add_argument(
    "--is_dpr",
    default=False,
    action="store_true",
    help="Patches have been saved in a disk patch representation folders format.",
)


#########################################################
group = parser.add_argument_group("Saving options")
#########################################################

parser.add_argument(
    "--max_n_wsis",
    default=None,
    type=int,
    help="Maximum number of WSIs to process. Mainly for debugging/prototyping.",
)


parser.add_argument(
    "--max_n_patches",
    default=None,
    type=int,
    help="Maximum number of patches per whole slide iamge to process.",
)

####################################
group = parser.add_argument_group("Model")
####################################

group.add_argument("--ckpt_path", type=str, help="Path to the MAE checkpoint file.")

group.add_argument(
    "--flatten", default=False, action="store_true", help="Flatten the model output."
)


####################################
group = parser.add_argument_group("Workflow")
####################################

group.add_argument("--device", default="auto", type=str, help="Which device to use.")

group.add_argument("--batch_size", default=512, type=int, help="Batch size to use.")


group.add_argument(
    "--guess_largest_batch_size",
    default=False,
    action="store_true",
    help="Guess the largest batch size we can fit on a single GPU.",
)

group.add_argument(
    "--continue_on_error_with_log",
    default=None,
    type=str,
    help="If None,then will not continue on error. Otherwise provide a directory to save error logs.",
)

group.add_argument(
    "--timeout_length",
    default=None,
    type=int,
    help="How long to wait for a single trial to finish (in seconds).",
)

args = parser.parse_args()

if args.is_dpr:
    raise NotImplementedError("TODO: add")

# Set the default timeout for Ray Tune trial execution
if args.timeout_length is not None:
    tune.stopper.TimeoutStopper.DEFAULT_GET_TIMEOUT = float(args.timeout_length)

print("Initializing Ray...")
ray.init()
print("Ray initialized")

n_avail_gpus = torch.cuda.device_count()
print("Found {} GPUs".format(n_avail_gpus))
print(ray.available_resources())

os.makedirs(args.save_dir, exist_ok=True)
process_info_fpath = os.path.join(args.save_dir, "processing_info.csv")

######################
# Identify WSI files #
######################

wsi_fpaths, bad_fpths_no_mpp = find_wsi_paths(
    folder=args.wsi_dir,
    recursive=False,
    skip_missing_mpp=True,
    max_n_wsis=args.max_n_wsis,
)

print("Found {} WSIs".format(len(wsi_fpaths)))
print(
    "{} WSIs were missing mpp values: {}".format(
        len(bad_fpths_no_mpp), bad_fpths_no_mpp
    )
)


wsi_fpaths, wsi_fpaths_missing_patch, _ = find_fpaths_with_matching_folder(
    wsi_fpaths=wsi_fpaths, super_folder=args.patch_dir
)

print(
    "Found {} WSIs with patches, {} missing patches,\n{}".format(
        len(wsi_fpaths), len(wsi_fpaths_missing_patch), wsi_fpaths_missing_patch
    )
)

##################
# Load the model #
##################

model = load_model(ckpt_path=args.ckpt_path)
model_ref = ray.put(model)


######################
# Guess batch size #
####################
if args.guess_largest_batch_size:
    # Load the patch size info from the first slide
    wsi_name = Path(wsi_fpaths[0]).stem
    wsi_patch_folder = os.path.join(args.patch_dir, wsi_name)
    wsi_patch_metadata = load_wsi_patch_metadata(folder=wsi_patch_folder)
    patch_size = wsi_patch_metadata["patch_size"]

    # create fake image
    img_shape = parse_patch_size(patch_size)
    x = np.random.normal(size=(3, *img_shape))

    start_time = time()
    best_batch_guess, try_info, other_info = guess_largest_batch_size(
        model=model,
        mode="extract",
        x=x,
        max_n_tries=10,
        batch_size_start=args.batch_size,
        max_batch_size=None,
        device_str=None,  # automatically detect gpu
        verbosity=1,
    )

    # save and print results
    try_info = pd.DataFrame(try_info)
    try_info.index.name = "try"
    try_info.to_csv(os.path.join(args.save_dir, "guess_larget_batch_size.csv"))
    guest_batch_size_runtime = time() - start_time
    print("Using a batch size of {}".format(best_batch_guess))
    print(
        "Guessing the best batch size took {} seconds".format(guest_batch_size_runtime)
    )

    if best_batch_guess is None:
        raise RuntimeError("Could not fit even one patch on the GPU")

    args.batch_size = best_batch_guess

#########################
# Extract for each WSI #
########################

if n_avail_gpus > 0:
    compute = ray.data.ActorPoolStrategy(size=n_avail_gpus)
    num_gpus_per_actor = 1

else:
    compute = ray.data.ActorPoolStrategy(size=1)
    num_gpus_per_actor = None

if args.continue_on_error_with_log is not None:
    os.makedirs(args.continue_on_error_with_log, exist_ok=True)
    print(
        'Continue on error protocol activated -- saving error logs to "{}"'.format(
            args.continue_on_error_with_log
        )
    )

for wsi_fpath in tqdm(wsi_fpaths, desc="WSI"):
    try:
        #########
        # Setup #
        #########

        # paths for saving + loading patch grid
        wsi_name = Path(wsi_fpath).stem  # file name without extensions
        wsi_feats_save_fpath = os.path.join(
            args.save_dir, "{}-patch_features.h5".format(wsi_name)
        )
        wsi_patch_folder = os.path.join(args.patch_dir, wsi_name)

        # load already computed patch grid
        patch_coords, patch_grid_idx = load_patch_grid_coords(
            folder=wsi_patch_folder, max_n=args.max_n_patches
        )

        wsi_patch_metadata = load_wsi_patch_metadata(folder=wsi_patch_folder)

        ################
        # Load patches #
        ################
        start_time = time()

        # TODO: add ability to stream this with a ray data source
        # e.g. perhaps read in the coordinates with ray.data.from_numpy(patches)
        patches = get_several_patches_at_mpp_ray(
            wsi_fpath=wsi_fpath,
            mpp=wsi_patch_metadata["mpp"],
            patch_size_mpp=wsi_patch_metadata["patch_size"],
            coords_level0=patch_coords,
            out_format="np",
        )

        load_patch_runtime = time() - start_time

        ####################
        # Extract features #
        ####################
        start_time = time()

        # ds = ray.data.from_numpy(patches) # DEPRECATED TODO remove this

        patches_lst = [patches[i] for i in range(patches.shape[0])]

        # Create a list of dictionaries with the key 'data'
        data_with_key = [{"data": patch} for patch in patches_lst]

        # Create the dataset using ray.data.from_items()
        ds = ray.data.from_items(data_with_key)

        transformed_ds = ds.map(NormalizeImageNp()) # TODO THIS LINE IS CAUSING ERROR

        output = transformed_ds.map_batches(
            fn=FeatureExtractor,
            fn_constructor_kwargs={"model_ref": model_ref, "device_str": args.device},
            batch_size=args.batch_size,
            compute=compute,
            num_gpus=num_gpus_per_actor,
            zero_copy_batch=True,
        ).take(len(patches))
        # write_numpy("local:{}".format(save_dir_this_wsi), column="features")

        features = np.stack([b["features"] for b in output])
        extract_feats_runtime = time() - start_time

        #########################
        # Save features to disk #
        #########################
        start_time = time()
        add_array_batch_to_h5(
            fpath=wsi_feats_save_fpath, data=features, name="features"
        )
        add_array_batch_to_h5(
            fpath=wsi_feats_save_fpath, data=patch_coords, name="coords"
        )
        add_array_batch_to_h5(
            fpath=wsi_feats_save_fpath, data=patch_grid_idx, name="patch_grid_idx"
        )

        with h5py.File(wsi_feats_save_fpath, "a") as file:
            for k, v in wsi_patch_metadata.items():
                file.attrs[k] = v

        save_feats_runtime = time() - start_time

        ###############################
        # Save processing information #
        ###############################
        process_info = {
            "wsi_name": wsi_name,
            "load_patch_runtime": load_patch_runtime,
            "extract_feats_runtime": extract_feats_runtime,
            "save_feats_runtime": save_feats_runtime,
            "finish_time": datetime.now().strftime("%Y-%m-%d__%H:%M:%S"),
            "n_patches": patch_coords.shape[0],
        }

        pd.DataFrame([process_info]).to_csv(
            process_info_fpath,
            mode="a",
            header=not os.path.exists(process_info_fpath),
            index=False,
        )

        print(process_info)

    except Exception as e:
        if args.continue_on_error_with_log is None:
            raise e
        else:
            os.makedirs(args.continue_on_error_with_log, exist_ok=True)
            # save an error log with the WSI name
            error_log_fpath = os.path.join(
                args.continue_on_error_with_log, "{}.txt".format(wsi_name)
            )
            with open(error_log_fpath, "w") as f:
                f.write(str(e))
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        break

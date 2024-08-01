#! /bin/bash

GPU_ID=1
DATA_ROOT_DIR="/ssd2/zhiwen/projects/InstantSplat/data"
DATASETS=(
    TT
    # MVimgNet
    )

SCENES=(
    # Barn
    Family
    # Francis
    # Horse
    # Ignatius
    )

N_VIEWS=(
    # 3
    # 5
    9
    # 12
    # 24
    )

# increase iteration to get better metrics (e.g. gs_train_iter=5000)
gs_train_iter=1000

for DATASET in "${DATASETS[@]}"; do
    for SCENE in "${SCENES[@]}"; do
        for N_VIEW in "${N_VIEWS[@]}"; do

            # Sparse_image_folder must be Absolute path
            Sparse_image_folder=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/24_views
            SOURCE_PATH=${Sparse_image_folder}/dust3r_${N_VIEW}_views
            MODEL_PATH=./output/eval/${DATASET}/${SCENE}/${N_VIEW}_views/
            GT_POSE_PATH=${DATA_ROOT_DIR}/Tanks_colmap/${SCENE}/24_views

            # ----- (1) Dust3r_coarse_geometric_initialization -----
            CMD_D1="CUDA_VISIBLE_DEVICES=${GPU_ID} python ./coarse_init_eval.py \
            --img_base_path ${Sparse_image_folder} \
            --n_views ${N_VIEW}  \
            --focal_avg \
            "

            # ----- (2) Train: jointly optimize pose -----
            CMD_T="CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train_joint.py \
            -s ${SOURCE_PATH} \
            -m ${MODEL_PATH}  \
            --n_views ${N_VIEW}  \
            --scene ${SCENE} \
            --iter ${gs_train_iter} \
            --optim_pose \
            "

            # ----- (3) Dust3r_test_pose_initialization -----
            CMD_D2="CUDA_VISIBLE_DEVICES=${GPU_ID} python ./init_test_pose.py \
            --img_base_path ${Sparse_image_folder} \
            --n_views ${N_VIEW}  \
            --focal_avg \
            "

            # ----- (4) Render -----
            CMD_R="CUDA_VISIBLE_DEVICES=${GPU_ID} python ./render.py \
            -s ${SOURCE_PATH} \
            -m ${MODEL_PATH}  \
            --n_views ${N_VIEW}  \
            --scene ${SCENE} \
            --optim_test_pose_iter 500 \
            --iter ${gs_train_iter} \
            --eval \
            "

            # ----- (5) Metrics -----
            CMD_M="CUDA_VISIBLE_DEVICES=${GPU_ID} python ./metrics.py \
            -m ${MODEL_PATH}  \
            --gt_pose_path ${GT_POSE_PATH} \
            --iter ${gs_train_iter} \
            --n_views ${N_VIEW}  \
            "

            echo "========= ${SCENE}: Dust3r_coarse_geometric_initialization ========="
            eval $CMD_D1
            echo "========= ${SCENE}: Train: jointly optimize pose ========="
            eval $CMD_T
            echo "========= ${SCENE}: Dust3r_test_pose_initialization ========="
            eval $CMD_D2
            echo "========= ${SCENE}: Render ========="
            eval $CMD_R
            echo "========= ${SCENE}: Metric ========="
            eval $CMD_M
            done
        done
    done
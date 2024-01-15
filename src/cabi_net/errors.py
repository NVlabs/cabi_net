# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


class BadPointCloudRendering(Exception):
    """Raised when there are no points rendered, probably due to bug in scene/camera placement"""

    pass


class VoxelOverflowError(Exception):
    """Raised when collision model has overflow, most probably because the voxel dimensions are not even"""

    pass


class ColllisionRatioError(Exception):
    """Still debugging why this is being raised"""

    pass


class SceneGenerationError(Exception):
    """Still debugging why this is being raised"""

    pass


class InfeasibleSceneError(Exception):
    """Raised when none of the poses in procedurally generated workspace trajectory has feasible kinematic IK solutions"""

    pass

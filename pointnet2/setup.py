from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="pointnet2",
    packages=["pointnet2"],
    package_dir={"pointnet2": "pointnet2"},
    install_requires=["torch", "tqdm", "plyfile"],
    ext_modules=[
        CUDAExtension(
            "pointnet2_cuda",
            [
                "pointnet2/src/pointnet2_api.cpp",
                "pointnet2/src/ball_query.cpp",
                "pointnet2/src/ball_query_gpu.cu",
                "pointnet2/src/group_points.cpp",
                "pointnet2/src/group_points_gpu.cu",
                "pointnet2/src/interpolate.cpp",
                "pointnet2/src/interpolate_gpu.cu",
                "pointnet2/src/sampling.cpp",
                "pointnet2/src/sampling_gpu.cu",
            ],
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

import os
from pathlib import Path
import setuptools
from torch_musa.utils.musa_extension import BuildExtension, MUSAExtension


if __name__ == '__main__':
    mtshmem_dir = os.getenv('MTSHMEM_DIR', None)
    assert mtshmem_dir is not None and os.path.exists(mtshmem_dir), 'Failed to find MTSHMEM'
    print(f'MTSHMEM directory: {mtshmem_dir}')

    # TODO: currently, we only support Hopper architecture, we may add Ampere support later
    os.environ['TORCH_MUSA_ARCH_LIST'] = '9.0'
    cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable',
                 '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes',
                 '-march=native', 'force_mcc']
    mtcc_flags = ['-O3', '-Xcompiler', '-O3', '-rdc=true', '-march=native']

    import torch_musa
    include_dirs = [
        Path(torch_musa.__file__.split("__init__")[0] + "share/torch_musa_codegen"),
        Path("/home/torch_musa"),
        'csrc/',
         f'{mtshmem_dir}/include',
    ]
    sources = ['csrc/deep_ep.cpp',
               'csrc/kernels/runtime.mu', 'csrc/kernels/intranode.mu',
               'csrc/kernels/internode.mu', 'csrc/kernels/internode_ll.mu']

    library_dirs = [f'{mtshmem_dir}/lib']
    # Disable aggressive MTX instructions
    if int(os.getenv('DISABLE_AGGRESSIVE_MTX_INSTRS', '0')):
        cxx_flags.append('-DDISABLE_AGGRESSIVE_MTX_INSTRS')
        mtcc_flags.append('-DDISABLE_AGGRESSIVE_MTX_INSTRS')

    # Disable DLTO (default by PyTorch)
    mtcc_dlink = ['-dlink', f'-L{mtshmem_dir}/lib', '-lmtshmem']
    extra_link_args = ['-l:libmtshmem.a', '-l:mtshmem_bootstrap_uid.so', f'-Wl,-rpath,{mtshmem_dir}/lib']
    extra_compile_args = {
        'cxx': cxx_flags,
        'mtcc': mtcc_flags,
        # disable device link
        # 'mtcc_dlink': mtcc_dlink  
    }

    # noinspection PyBroadException
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception as _:
        revision = ''

    setuptools.setup(
        name='deep_ep',
        version='1.0.0' + revision,
        packages=setuptools.find_packages(
            include=['deep_ep']
        ),
        ext_modules=[
            MUSAExtension(
                name='deep_ep_cpp',
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                sources=sources,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )

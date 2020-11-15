from setuptools import setup, find_packages
import pip
import logging
import pkg_resources


def _parse_requirements(file_path):
    pip_ver = pkg_resources.get_distribution('pip').version
    pip_version = list(map(int, pip_ver.split('.')[:2]))
    if pip_version >= [6, 0]:
        raw = pip.req.parse_requirements(file_path,
                                         session=pip.download.PipSession())
    else:
        raw = pip.req.parse_requirements(file_path)
    return [str(i.req) for i in raw]


try:
    install_reqs = _parse_requirements("requirements.txt")
except Exception:
    logging.warning('Fail load requirements file, so using default ones.')
    install_reqs = []

setup(
    name="macroecon-default",
    version="0.1.0",
    url="https://github.com/samsonq/Macroeconomic-Default-Analysis",
    description="",
    install_requires=install_reqs,
    python_requires=">=3.4",
    keywords=""
)
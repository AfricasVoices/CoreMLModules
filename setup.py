from setuptools import setup

setup(
    name="CoreMLModules",
    version="0.1.0",
    url="https://github.com/AfricasVoices/CoreMLModules",
    packages=["core_data_modules"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest<=3.6.4"]
)

from setuptools import setup, Distribution, Extension
class BinaryDistribution(Distribution):
    def has_ext_modules(foo):
        return True

long_description = "deploy_paddle is a toolkit for converting trained model to deploy in different platform.\n\n"

long_description += "Email: dltp-sz@baidu.com"

setup(
    name="paddle2lite",
    version='0.0.1',
    author="dltp-sz",
    author_email="dltp-sz@baidu.com",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="",
    packages=['paddle2lite'],
    package_data={
                    'paddle2lite': ['lite.so'],
                            },
    package_dir={'': '.'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    #python_requires='==3.6.*',
    license='Apache 2.0')

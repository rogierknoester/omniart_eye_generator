import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(name='omniart_eye_generator',
                 version='0.1',
                 description='A cDCGAN that can generate eyes',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url='http://github.com/rogierknoester/omniart_eye_generator',
                 author='Rogier Knoester',
                 author_email='knoesterrogier+omniart@gmail.com',
                 license='MIT',
                 packages=setuptools.find_packages(),
                 package_data={'omniart_eye_generator': ['*.pth']},
                 zip_safe=False)

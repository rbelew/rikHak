from distutils.core import setup
setup(name='json2djf',
      version=json2djf.__version__,
      description='A script to construct a data fixture suitable for import by django from a "standard" JSON data file',
      author='R. K. Belew',
      author_email='rik@electronicArtifacts.com',
      py_modules=['json2djf'],
      )

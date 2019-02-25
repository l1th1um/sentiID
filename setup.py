from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='sentiID',
      version='1.0.3',
      description='NLP Toolkit for Indonesian Languange - Sentiment Analysis',
      long_description=readme(),
      url='http://git.informatika.lipi.go.id/gitlab/idnlp',
      author='p2i',
      author_email='info@informatika.lipi.go.id',
      license='MIT',
      packages=find_packages(),
      package_data={'sentiID': ['sentiID/data/*.*']},
      include_package_data = True,
      install_requires=[
          'numpy==1.13.3',
          'pandas==0.21.0',
          'python-dateutil==2.6.1',
          'pytz==2017.3',
          'scikit-learn==0.19.0',
          'scipy==1.0.0',
          'six==1.11.0',
          'sklearn==0.0'
      ],
      zip_safe=False)

from setuptools import setup

setup(
    name = 'maq',
    packages = ['maq'],
    version = '0.2',  # Ideally should be same as your GitHub release tag varsion
    description = 'Machine Learning on Adiabatic Quantum Computers (MAQ) is a library of algorithms used to train machine learning models on adiabatic quantum computers.',
    long_description = 'Machine Learning on Adiabatic Quantum Computers (MAQ) is a library of algorithms used to train machine learning models on adiabatic quantum computers.',
    long_description_content_type = 'text/markdown',
    author = 'Prasanna Date, Kathleen Hamilton, Robert Patton, Travis Humble, Thomas Potok',
    author_email = 'datepa@ornl.gov',
    include_package_data=True,
    project_urls = {"Source": "https://github.com/prasannadate/maq"},
    url = 'https://github.com/prasannadate/maq',
    download_url = 'https://github.com/prasannadate/maq/archive/refs/tags/v1.0.0.tar.gz',
    keywords = ['Adiabatic Quantum Machine Learning', 'Quantum Machine Learning', 'Adiabatic Quantum Computing', 'Quantum Computing', 'Machine Learning'],
    license = 'BSD License',
    install_requires = ["numpy"],
    classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: BSD License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
  ]
)

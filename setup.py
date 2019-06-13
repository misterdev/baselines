import os

from setuptools import setup, find_packages

# TODO: setup does not support installation from url, move to requirements*.txt
# TODO: @master as soon as mr is merged on flatland.
os.system('pip install git+https://gitlab.aicrowd.com/flatland/flatland.git@57-access-resources-through-importlib_resources')

install_reqs = []
# TODO: include requirements_RLLib_training.txt
requirements_paths = ['requirements_torch_training.txt'] #, 'requirements_RLLib_training.txt']
for requirements_path in requirements_paths:
    with open(requirements_path, 'r') as f:
        install_reqs += [
            s for s in [
                line.strip(' \n') for line in f
            ] if not s.startswith('#') and s != ''
        ]
requirements = install_reqs
setup_requirements = install_reqs
test_requirements = install_reqs

setup(
    author="S.P. Mohanty",
    author_email='mohanty@aicrowd.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Multi Agent Reinforcement Learning on Trains",
    entry_points={
        'console_scripts': [
            'flatland=flatland.cli:main',
        ],
    },
    install_requires=requirements,
    long_description='',
    include_package_data=True,
    keywords='flatland-baselines',
    name='flatland-rl-baselines',
    packages=find_packages('.'),
    data_files=[],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://gitlab.aicrowd.com/flatland/baselines',
    version='0.1.1',
    zip_safe=False,
)

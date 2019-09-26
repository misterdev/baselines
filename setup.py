from setuptools import setup, find_packages

install_reqs = []
dependency_links = []
requirements_paths = ['requirements_torch_training.txt']
for requirements_path in requirements_paths:
    with open(requirements_path, 'r') as f:
        install_reqs += [
            s for s in [
                line.strip(' \n') for line in f
            ] if not s.startswith('#') and s != '' and not s.startswith('git+')
        ]
with open(requirements_path, 'r') as f:
    dependency_links += [
        s for s in [
            line.strip(' \n') for line in f
        ] if s.startswith('git+')
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
    dependency_links=dependency_links,
    url='https://gitlab.aicrowd.com/flatland/baselines',
    version='0.1.1',
    zip_safe=False,
)

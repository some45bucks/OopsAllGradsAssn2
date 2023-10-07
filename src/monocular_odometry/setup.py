import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'monocular_odometry'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'MCN = monocular_odometry.MCN:main',
            'DRAC = monocular_odometry.DRAC:main',
            'IMU = monocular_odometry.IMU:main',
            'log = monocular_odometry.log:main'
        ],
    },
)
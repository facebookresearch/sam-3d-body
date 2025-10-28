pose_info = dict(
    pose_format='openpose',
    paper_info=dict(
        author='Zhe, Cao and Tomas, Simon and '
        'Shih-En, Wei and Yaser, Sheikh',
        title='OpenPose: Realtime Multi-Person 2D Pose '
        'Estimation using Part Affinity Fields',
        container='IEEE Transactions on Pattern Analysis '
        'and Machine Intelligence',
        year='2019',
        homepage='https://github.com/CMU-Perceptual-Computing-Lab/openpose/',
    ),
    keypoint_info={
        0:
        dict(name='nose', id=0, color=[255, 0, 0], type='upper', swap=''),
        1:
        dict(name='neck', id=1, color=[255, 85, 0], type='upper', swap=''),
        2:
        dict(
            name='right_shoulder',
            id=2,
            color=[255, 170, 0],
            type='upper',
            swap='left_shoulder'),
        3:
        dict(
            name='right_elbow',
            id=3,
            color=[255, 255, 0],
            type='upper',
            swap='left_elbow'),
        4:
        dict(
            name='right_wrist',
            id=4,
            color=[170, 255, 0],
            type='upper',
            swap='left_wrist'),
        5:
        dict(
            name='left_shoulder',
            id=5,
            color=[85, 255, 0],
            type='upper',
            swap='right_shoulder'),
        6:
        dict(
            name='left_elbow',
            id=6,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        7:
        dict(
            name='left_wrist',
            id=7,
            color=[0, 255, 85],
            type='upper',
            swap='right_wrist'),
        8:
        dict(
            name='mid_hip',
            id=8,
            color=[0, 255, 170],
            type='lower',
            swap=''),
        9:
        dict(
            name='right_hip',
            id=9,
            color=[0, 255, 255],
            type='lower',
            swap='left_hip'),
        10:
        dict(
            name='right_knee',
            id=10,
            color=[0, 170, 255],
            type='lower',
            swap='left_knee'),
        11:
        dict(
            name='right_ankle',
            id=11,
            color=[0, 85, 255],
            type='lower',
            swap='left_ankle'),
        12:
        dict(
            name='left_hip',
            id=12,
            color=[0, 0, 255],
            type='lower',
            swap='right_hip'),
        13:
        dict(
            name='left_knee',
            id=13,
            color=[85, 0, 255],
            type='lower',
            swap='right_knee'),
        14:
        dict(
            name='left_ankle',
            id=14,
            color=[170, 0, 255],
            type='lower',
            swap='right_ankle'),
        15:
        dict(
            name='right_eye',
            id=15,
            color=[255, 0, 255],
            type='upper',
            swap='left_eye'),
        16:
        dict(
            name='left_eye',
            id=16,
            color=[255, 0, 170],
            type='upper',
            swap='right_eye'),
        17:
        dict(
            name='right_ear',
            id=17,
            color=[255, 0, 85],
            type='upper',
            swap='left_ear'),
        18:
        dict(
            name='left_ear',
            id=18,
            color=[255, 0, 0],
            type='upper',
            swap='right_ear'),
        19:
        dict(
            name='left_big_toe',
            id=19,
            color=[255, 170, 80],
            type='lower',
            swap='right_big_toe'),
        20:
        dict(
            name='left_small_toe',
            id=20,
            color=[255, 170, 170],
            type='lower',
            swap='right_small_toe'),
        21:
        dict(
            name='left_heel',
            id=21,
            color=[255, 170, 255],
            type='lower',
            swap='right_heel'),
        22:
        dict(
            name='right_big_toe',
            id=22,
            color=[255, 0, 170],
            type='lower',
            swap='left_big_toe'),
        23:
        dict(
            name='right_small_toe',
            id=23,
            color=[255, 85, 170],
            type='lower',
            swap='left_small_toe'),
        24:
        dict(
            name='right_heel',
            id=24,
            color=[255, 170, 170],
            type='lower',
            swap='left_heel'),
    },
    skeleton_info={
        0: dict(link=('neck', 'right_shoulder'), id=0, color=[255, 0, 0]),
        1: dict(link=('neck', 'left_shoulder'), id=1, color=[255, 85, 0]),
        2: dict(
            link=('right_shoulder', 'right_elbow'), id=2, color=[255, 170, 0]),
        3:
        dict(link=('right_elbow', 'right_wrist'), id=3, color=[255, 255, 0]),
        4:
        dict(link=('left_shoulder', 'left_elbow'), id=4, color=[170, 255, 0]),
        5: dict(link=('left_elbow', 'left_wrist'), id=5, color=[85, 255, 0]),
        6: dict(link=('neck', 'mid_hip'), id=6, color=[255, 0, 85]),
        7: dict(link=('right_hip', 'right_knee'), id=7, color=[0, 255, 85]),
        8: dict(link=('right_knee', 'right_ankle'), id=8, color=[0, 255, 170]),
        9: dict(link=('left_hip', 'left_knee'), id=9, color=[0, 170, 255]),
        10: dict(link=('left_knee', 'left_ankle'), id=10, color=[0, 85, 255]),
        11: dict(link=('neck', 'nose'), id=12, color=[0, 0, 255]),
        12: dict(link=('nose', 'right_eye'), id=12, color=[255, 0, 170]),
        13: dict(link=('right_eye', 'right_ear'), id=13, color=[170, 0, 255]),
        14: dict(link=('nose', 'left_eye'), id=14, color=[255, 0, 255]),
        15: dict(link=('left_eye', 'left_ear'), id=15, color=[255, 0, 170]),
        16: dict(link=('mid_hip', 'right_hip'), id=16, color=[255, 85, 85]),
        17: dict(link=('mid_hip', 'left_hip'), id=17, color=[255, 85, 0]),
        18: dict(link=('left_ankle', 'left_big_toe'), id=18, color=[85, 255, 255]),
        19: dict(link=('left_big_toe', 'left_small_toe'), id=19, color=[85, 255, 170]),
        20: dict(link=('left_ankle', 'left_heel'), id=20, color=[255, 255, 85]),
        21: dict(link=('right_ankle', 'right_big_toe'), id=21, color=[170, 255, 85]),
        22: dict(link=('right_big_toe', 'right_small_toe'), id=22, color=[85, 255, 85]),
        23: dict(link=('right_ankle', 'right_heel'), id=23, color=[0, 255, 0]),
    },
    joint_weights=[1.] * 25,
    sigmas=[
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.082, 0.082, 0.026,
        0.025, 0.025, 0.026, 0.025, 0.025
    ])

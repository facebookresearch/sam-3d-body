pose_info = dict(
    pose_format='joint14',  # 14 joints that are commonly used for 3D evaluation
    keypoint_info={
        0:
        dict(
            name='right_ankle',
            id=0,
            color=[255, 128, 0],
            type='lower',
            swap='left_ankle'),
        1:
        dict(
            name='right_knee',
            id=1,
            color=[255, 128, 0],
            type='lower',
            swap='left_knee'),
        2:
        dict(
            name='right_hip',
            id=2,
            color=[255, 128, 0],
            type='lower',
            swap='left_hip'),
        3:
        dict(
            name='left_hip',
            id=3,
            color=[0, 255, 0],
            type='lower',
            swap='right_hip'),
        4:
        dict(
            name='left_knee',
            id=4,
            color=[0, 255, 0],
            type='lower',
            swap='right_knee'),
        5:
        dict(
            name='left_ankle',
            id=5,
            color=[0, 255, 0],
            type='lower',
            swap='right_ankle'),
        6:
        dict(
            name='right_wrist',
            id=6,
            color=[255, 128, 0],
            type='upper',
            swap='left_wrist'),
        7:
        dict(
            name='right_elbow',
            id=7,
            color=[255, 128, 0],
            type='upper',
            swap='left_elbow'),
        8:
        dict(
            name='right_shoulder',
            id=8,
            color=[255, 128, 0],
            type='upper',
            swap='left_shoulder'),
        9:
        dict(
            name='left_shoulder',
            id=9,
            color=[0, 255, 0],
            type='upper',
            swap='right_shoulder'),
        10:
        dict(
            name='left_elbow',
            id=10,
            color=[0, 255, 0],
            type='upper',
            swap='right_elbow'),
        11:
        dict(
            name='left_wrist',
            id=11,
            color=[0, 255, 0],
            type='upper',
            swap='right_wrist'),
        12:
        dict(name='neck_(LSP)', id=12, color=[51, 153, 255], type='upper', swap=''),
        13:
        dict(
            name='head_(H36M)',
            id=13,
            color=[0, 0, 255],
            type='upper',
            swap=''),
    },
    skeleton_info={
        0:
        dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1:
        dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2:
        dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        3:
        dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        4:
        dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        5:
        dict(link=('head_(H36M)', 'neck_(LSP)'), id=12, color=[51, 153, 255]),
        6:
        dict(link=('left_shoulder', 'left_hip'), id=6, color=[128, 153, 255]),
        7:
        dict(link=('right_shoulder', 'right_hip'), id=7, color=[0, 153, 255]),
        8:
        dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        9:
        dict(
            link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        10:
        dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11:
        dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12:
        dict(link=('neck_(LSP)', 'left_shoulder'), id=12, color=[51, 153, 128]),
        13:
        dict(
            link=('neck_(LSP)', 'right_shoulder'), id=13, color=[51, 153, 128])
    },
    joint_weights=[1.] * 14,
    sigmas=[0.025] * 14,
)
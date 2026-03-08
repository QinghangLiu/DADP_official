import numpy as np

def generate_custom_task_walker(task):
    # dims 4-6 (index 4,5,6):
    mass = np.random.uniform(2.99, 6.69)
    task[4] = mass
    task[5] = mass
    task[6] = mass
    # dims 7-10 (index 7,8,9,10):
    low = np.array([0.3, 0.27, 0.27, 0.3])
    size = np.random.uniform(1,2)
    task[7:11] = low * size
    # dims 11-12 (index 11,12):
    friction = np.random.uniform(1.32, 2.28)
    task[11] = friction
    task[12] = friction
    return task

def generate_halfcheetah_task(task):
    # Randomize dim 7
    task[7] = np.random.uniform(0.3364, 0.9514)
    # Randomize dim 8
    task[8] = np.random.uniform(0.0841, 0.2378)
    # Randomize dims (12,13,14) as a group
    group_12_13_14 = np.random.uniform(1,2.5)
    task[12:15] = group_12_13_14 * np.array([0.0841, 0.1, 0.0748])
    # Other dims: sample uniformly in [0,1]
    return task

def generate_ant_task(task):
    task[14] = np.random.uniform(0.4653, 1.6119)
    task[15] = np.random.uniform(0.4653, 1.6119)
    task[16] = np.random.uniform(0.4653, 1.6119)
    task[17] = np.random.uniform(0.4653, 1.6119)
    return task

def generate_hopper_task(task):

    task[5] = np.random.uniform(0.75, 1.19)
    task[6] = np.random.uniform(0.75, 1.19)
    task[7] = np.random.uniform(0.75, 1.19)
    return task
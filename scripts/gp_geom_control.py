import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import mujoco
import glfw
import scipy.interpolate as si
from scipy.spatial.transform import Rotation
from assets.logger import Logger
from assets.util import sync, quat_conj, quat_mult
from ctrl.GeomControl import GeomControl
from ctrl.RobustGPGeomControl import RobustGPGeomControl
from ctrl.GPGeomControl import GPGeomControl


def run_simulation(controller_type):
    # Reading model data
    model = mujoco.MjModel.from_xml_path("../assets/cf2.xml")
    data = mujoco.MjData(model)

    # Initialize the library
    if not glfw.init():
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(1280, 720, "Crazyflie in MuJoCo", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)

    # initialize visualization data structures
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation = 170, -30
    cam.lookat, cam.distance = [0, 0, 1], 2

    pert = mujoco.MjvPerturb()
    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(model, maxgeom=30)
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

    ## To obtain inertia matrix
    mujoco.mj_step(model, data)

    ## Controller
    if controller_type == 1:
        controller = GeomControl(model, data)
    elif controller_type == 2:
        controller = RobustGPGeomControl(model, data, train_x=train_x, train_y=train_y)
        controller.train_model()
        controller.compute_delta()
    elif controller_type == 3:
        controller = GPGeomControl(model, data, train_x=train_x, train_y=train_y)
        controller.train_model()
    else:
        raise ValueError

    ## Trajectory
    pos_ref_points = np.loadtxt("../assets/pos_ref.csv", delimiter=',')
    pos_ref_time = np.linspace(0, 0.9, pos_ref_points.shape[0])
    flip_traj = [si.splrep(pos_ref_time, pos_ref_points[:, 0], k=3),
                 si.splrep(pos_ref_time, 0 * pos_ref_points[:, 0], k=3),
                 si.splrep(pos_ref_time, pos_ref_points[:, 1], k=3)]
    quat_points = -1.99999 / (1 + np.exp(-20*(pos_ref_time-0.9/2))) + 1.99999/2
    rot_traj = si.splrep(pos_ref_time, quat_points, k=3)

    # eval_t = np.linspace(0, 0.9, 500)
    # q0 = si.splev(eval_t, rot_traj)
    # q2 = np.sqrt(1 - q0 ** 2)
    # target_quat = np.vstack([np.zeros_like(q0), q2, np.zeros_like(q2), q0]).T
    # dq0 = si.splev(eval_t, rot_traj, der=1)
    # dq2 = - dq0 * q0 / q2
    # target_quat_vel = np.vstack([np.zeros_like(dq0), dq2, np.zeros_like(dq2), dq0]).T
    # target_eul = np.zeros((500, 3))
    # for i in range(500):
    #     target_eul[i, :] = Rotation.from_quat((np.roll(target_quat[i, :], -1))).as_euler('xyz')
    # target_ang_vel = np.zeros_like(target_eul)
    # for i in range(500):
    #     target_ang_vel[i, :] = (2 * quat_mult(quat_conj(target_quat[i, :]), target_quat_vel[i, :]))[0:3]
    # # plt.figure()
    # # plt.plot(target_quat)
    # plt.figure()
    # plt.plot(target_quat)
    # plt.figure()
    # plt.plot(target_quat_vel)
    # plt.show()



    target_quat = np.array([1, 0, 0, 0])
    target_ang_vel = np.zeros(3)
    psi = 0
    eta_R, mu_R, e_R = np.zeros(3), np.zeros(3), np.zeros(3)

    timestep = 0.005
    simulation_step = 0.005
    graphics_step = 0.02

    episode_length = 0.99
    logger = Logger(episode_length, control_step)

    start = time.time()

    for i in range(int(episode_length / control_step)):
        simtime = data.time

        if simtime < 0.1:
            target_pos = np.array([0, 0, 1])
            pos = data.qpos[0:3]
            quat = data.qpos[3:7]
            vel = data.qvel[0:3]
            ang_vel = data.qvel[3:6]
            data.ctrl = controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos)

        elif simtime < 1:
            eval_time = (simtime - 0.1)
            target_pos = np.array([si.splev(eval_time, flip_traj[i]) for i in range(3)])
            target_pos[2] = target_pos[2] + 1
            target_vel = np.array([si.splev(eval_time, flip_traj[i], der=1) for i in range(3)])
            target_acc = np.array([si.splev(eval_time, flip_traj[i], der=2) for i in range(3)])
            q0 = si.splev(eval_time, rot_traj)
            q2 = np.sqrt(1 - q0 ** 2)
            target_quat = np.array([q0, 0, q2, 0])
            dq0 = si.splev(eval_time, rot_traj, der=1)
            dq2 = - dq0 * q0 / q2
            target_quat_vel = np.array([dq0, 0, dq2, 0])
            target_ang_vel = np.roll((2 * quat_mult(quat_conj(np.roll(target_quat, -1)), np.roll(target_quat_vel, -1)))[0:3], -1)

            pos = data.qpos[0:3]
            quat = data.qpos[3:7]
            vel = data.qvel[0:3]
            ang_vel = data.qvel[3:6]
            data.ctrl, eta_R, mu_R, psi, e_R = controller.compute_att_control(pos, quat, vel, ang_vel, target_pos, target_vel, target_acc,
                                                       target_quat=target_quat, target_ang_vel=target_ang_vel) # target_quat_vel=target_quat_vel)
        else:
            target_pos = np.array([0, 0, 1])
            pos = data.qpos[0:3]
            quat = data.qpos[3:7]
            vel = data.qvel[0:3]
            ang_vel = data.qvel[3:6]
            data.ctrl = controller.compute_pos_control(pos, quat, vel, ang_vel, target_pos)

        data.xfrc_applied[1, 3:6] = np.array([-0.008, -0.008, 0])*(quat[2])

        for _ in range(int(control_step / simulation_step)):
            mujoco.mj_step(model, data, 1)
        state = np.hstack([target_pos, pos, Rotation.from_quat((np.roll(quat, -1))).as_euler('xyz'),
                           Rotation.from_quat((np.roll(target_quat, -1))).as_euler('xyz'), ang_vel, target_ang_vel,
                           quat, psi, e_R])
        gp_terms = np.hstack([eta_R[0:2], mu_R[0:2]])
        logger.log(timestamp=simtime, state=state, control=data.ctrl, gp_terms=gp_terms)

        if i % (graphics_step / control_step) == 0:
            viewport = mujoco.MjrRect(0, 0, 0, 0)
            viewport.width, viewport.height = glfw.get_framebuffer_size(window)
            mujoco.mjv_updateScene(model, data, opt, pert=None, cam=cam, catmask=mujoco.mjtCatBit.mjCAT_ALL, scn=scn)
            mujoco.mjr_render(viewport, scn, con)

            glfw.swap_buffers(window)
            glfw.poll_events()

            # sync with wall-clock time
            sync(i, start, timestep*5)

            if glfw.window_should_close(window):
                break

    glfw.terminate()
    return logger


if __name__ == "__main__":

    control_step = 0.005

    # Run simulation with nominal geometric control
    logger_nom = run_simulation(1)
    # logger.save_as_csv("nominal_dist")

    # Extract training inputs and targets
    torque = logger_nom.controls[0, 1:, :]
    euler = logger_nom.states[0, 6:9, :]
    q = logger_nom.states[0, 18:22, :]

    l = 0.046 / np.sqrt(2)

    inertia = np.diag([1.4e-5, 1.4e-5, 2.17e-5])

    ang_vel = logger_nom.states[0, 12:15, :]
    target_ang_vel = logger_nom.states[0, 15:18, :]
    ang_accel = np.zeros_like(ang_vel)

    for i in range(3):
        ang_accel[i, :] = np.gradient(ang_vel[i, :], control_step)
    delta_R = np.zeros_like(ang_accel)
    for i in range(ang_accel.shape[1]):
        delta_R[:, i] = inertia @ ang_accel[:, i] + np.cross(ang_vel[:, i], inertia @ ang_vel[:, i]) - torque[:, i]

    start = int(0.14 / control_step)
    end = int(0.89 / control_step)

    # plt.figure()
    # plt.plot(delta_R[0, :])
    # plt.plot(delta_R[1, :])
    # plt.plot(logger_nom.states[0, 7, start:end].T)
    # plt.plot(logger_nom.states[0, 10, start:end].T)
    # plt.figure()
    # plt.plot(target_ang_vel[:, start:end].T)
    # plt.plot(ang_vel[1, start:end])
    plt.figure()
    plt.plot(logger_nom.states[0, 22, start:end])
    # plt.figure()
    # plt.plot(q[2, start:end], delta_R[1, start:end])
    # plt.legend(('x', 'y'))
    # plt.figure()
    # plt.plot(torque[:, start:end].T)
    plt.show(block=False)

    # start = int(0.11 / control_step)
    # end = int(0.89 / control_step)
    train_step = 3

    # start = int(1*ARGS.control_freq_hz)
    # end = int(4*ARGS.control_freq_hz)
    # train_step = 20
    train_x = q[2, start:end]
    # train_x = np.hstack((q[start:end, 0:2], ang_vel[0:2, start:end].T))
    train_y = [delta_R[0, start:end], delta_R[1, start:end]]

    train_x = train_x[::train_step]
    train_y = [train_yi[::train_step]*200 + 0.01*np.random.randn(train_yi[::train_step].shape[0]) for train_yi in train_y]
    train_x = torch.from_numpy(train_x).float()
    train_y = [torch.from_numpy(train_y[i]).float() for i in range(2)]

    logger_gp_mean = run_simulation(3)
    # logger.save_as_csv("gp_mean_dist")
    plt.figure()
    plt.plot(logger_gp_mean.states[0, 22, start:end])
    plt.show(block=False)

    logger_gp_rob = run_simulation(2)
    # logger.save_as_csv("gp_robust_dist")
    plt.figure()
    plt.plot(logger_gp_rob.states[0, 22, start:end])
    plt.show(block=False)

    fig, axs = plt.subplots(6, 2)
    logger_nom.plot(False, fig, axs)
    logger_gp_mean.plot(False, fig, axs)
    logger_gp_rob.plot(True, fig, axs)

    # logger_nom.save_log('simu_nom.csv')
    # logger_gp_mean.save_log('simu_ada.csv')
    # logger_gp_rob.save_log('simu_rob.csv')

import time
import mujoco
import numpy as np
from bayes_opt import BayesianOptimization
from ctrl.FeedforwardBackflip import FeedforwardBackflip
from scipy.spatial.transform import Rotation


def compute_final_error(data):
    return np.array([data.qpos[0], data.qpos[2]-1, 0.2 * data.qvel[0], 0.2 * data.qvel[2],
                          Rotation.from_quat(np.roll(data.qpos[3:7], -1)).as_euler('xyz')[1]])


def simulate(params):  #(U1, T1, T3, U5, T5):
    # params = (U1, T1, T3, U5, T5)
    F, tau = flip.compute_control_sequence(params)
    episode_length = F.shape[0]
    data.qpos = np.array([0, 0, 1, 1, 0, 0, 0])
    data.qvel = np.zeros(6)
    for i in range(episode_length):
        data.ctrl = np.array([F[i], 0, tau[i], 0])
        for _ in range(int(control_step / simulation_step)):
            mujoco.mj_step(model, data, 1)
    final_error = compute_final_error(data)
    return np.linalg.norm(final_error)


def test(params):
    graphics_step = 0.02
    import glfw
    from assets.util import sync
    from assets.logger import Logger

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
    F, tau = flip.compute_control_sequence(params)
    episode_length = F.shape[0]
    logger = Logger(episode_length*control_step, control_step)
    start = time.time()
    for i in range(episode_length):
        simtime = data.time
        data.ctrl = np.array([F[i], 0, tau[i], 0])
        for _ in range(int(control_step / simulation_step)):
            mujoco.mj_step(model, data, 1)
        state = np.hstack([data.qpos[0:3], data.qvel[0:3],
                           Rotation.from_quat((np.roll(data.qpos[3:7], -1))).as_euler('xyz'), data.qvel[3:6],
                           np.zeros(14)])
        logger.log(timestamp=simtime, state=state, control=data.ctrl)
        if i % (graphics_step / control_step) == 0:
            viewport = mujoco.MjrRect(0, 0, 0, 0)
            viewport.width, viewport.height = glfw.get_framebuffer_size(window)
            mujoco.mjv_updateScene(model, data, opt, pert=None, cam=cam, catmask=mujoco.mjtCatBit.mjCAT_ALL, scn=scn)
            mujoco.mjr_render(viewport, scn, con)

            glfw.swap_buffers(window)
            glfw.poll_events()

            # sync with wall-clock time
            sync(i, start, control_step*10)

            if glfw.window_should_close(window):
                break
    final_error = compute_final_error(data)
    print(final_error)
    print(np.linalg.norm(final_error))
    logger.plot_ff()


if __name__ == '__main__':
    model = mujoco.MjModel.from_xml_path("../assets/cf2.xml")
    data = mujoco.MjData(model)
    flip = FeedforwardBackflip(model, data)
    control_step = 0.005
    simulation_step = 0.005

    test([17.8, 0.15, 0.2, 17.8, 0.12])
    # test([17.728831803463013, 0.22, 0.34420808748010356, 17.704231994964093, 0.22])  # 0.06470557974463498
    # test([15.0, 0.09427028163588136, 0.05, 15.298548456166605, 0.11526160336510613])
    #  fun: 0.1552123771622505
    #             x: [15.0, 0.09427028163588136, 0.05, 15.298548456166605, 0.11526160336510613]
    # simulate(12.55, 0.0737, 0.2973, 15.52, 0.1)

    # pbounds = {'U1': (12, 17.85), 'T1': (0.05, 0.2), 'T3': (0.1, 0.4), 'U5': (12, 17.85), 'T5': (0.1, 0.4)}
    # optimizer = BayesianOptimization(
    #     f=simulate,
    #     pbounds=pbounds,
    #     verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    #     random_state=10,
    # )
    # # load_logs(optimizer, logs=["./logs-init-400-iter-1000.json"]);
    #
    # optimizer.probe(
    #     params=[17.5, 0.12, 0.3, 17.5, 0.3],
    #     lazy=True,
    # )
    # optimizer.probe(
    #     params=[17.5, 0.2, 0.35, 17.5, 0.1],
    #     lazy=True,
    # )
    #
    # optimizer.maximize(
    #     init_points=10,
    #     n_iter=1500,
    #     acq="ucb",
    # )
    # print(optimizer.max)
    # test((17.5, 0.2, 0.35, 17.5, 0.2))

    # from skopt import gp_minimize
    #
    # res = gp_minimize(simulate,  # the function to minimize
    #                   [(15, 17.85), (0.05, 0.3), (0.05, 0.4), (15, 17.85),  (0.05, 0.3)],
    #                   acq_func="LCB",  # the acquisition function
    #                   n_calls=500,  # the number of evaluations of f
    #                   n_random_starts=10,  # the number of random initialization points
    #                   noise=0.01 ** 2,  # the noise level (optional)
    #                   random_state=1234,  # the random seed
    #                   verbose=True)
    # print(res)
    # from skopt.plots import plot_convergence
    # plot_convergence(res)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.show()

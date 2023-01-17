import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os


class Logger:
    """A class for logging and visualization.

    Stores, saves to file, and plots the kinematic information and RPMs
    of a simulation with one or more drones.

    """

    ################################################################################

    def __init__(self,
                 episode_length,
                 timestep,
                 num_runs=1
                 ):
        self.episode_steps = int(episode_length / timestep)
        self.timestamps = np.zeros((num_runs, self.episode_steps))
        #### Note: this is the suggest information to log ##############################
        self.states = np.zeros((num_runs, 26, self.episode_steps))  #### 13 states: pos_x,
        # pos_y,
        # pos_z,
        # vel_x,
        # vel_y,
        # vel_z,
        # quat_w,
        # quat_x,
        # quat_y,
        # quat_z,
        # ang_vel_x,
        # ang_vel_y,
        # ang_vel_z,

        #### Note: this is the suggest information to log ##############################
        self.controls = np.zeros((num_runs, 4, self.episode_steps))  #### 4 ctrl inputs
        self.gp_terms = np.zeros((num_runs, 4, self.episode_steps))  #### 4 ctrl inputs
        self.current_step = 0

    ################################################################################

    def reset_counter(self):
        self.current_step = 0

    ################################################################################

    def log(self,
            timestamp,
            state,
            control=np.zeros(4),
            gp_terms=np.zeros(4),
            current_run=0
            ):
        #### Log the information and increase the counter ##########
        if self.current_step > self.episode_steps:
            print('Somethings wrong in the logging department')
        else:
            self.timestamps[current_run, self.current_step] = timestamp
            self.states[current_run, :, self.current_step] = state
            self.controls[current_run, :, self.current_step] = control
            self.gp_terms[current_run, :, self.current_step] = gp_terms
            self.current_step = self.current_step + 1

    ################################################################################

    def save(self):
        """Save the logs to file.
        """
        with open(
                os.path.dirname(os.path.abspath(__file__)) + "/../../files/logs/save-flight-" + datetime.now().strftime(
                        "%m.%d.%Y_%H.%M.%S") + ".npy", 'wb') as out_file:
            np.savez(out_file, timestamps=self.timestamps, states=self.states, controls=self.controls)

    ################################################################################

    def save_as_csv(self,
                    comment: str = ""
                    ):
        """Save the logs---on your Desktop---as comma separated values.

        Parameters
        ----------
        comment : str, optional
            Added to the foldername.

        """
        csv_dir = os.path.dirname(
            os.path.abspath(__file__)) + "/../files/logs/save-flight-" + comment  # + "-" + datetime.now().strftime(
        # "%m.%d.%Y_%H.%M.%S")
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir + '/')
        t = np.arange(0, self.timestamps.shape[1] / self.LOGGING_FREQ_HZ, 1 / self.LOGGING_FREQ_HZ)
        for i in range(self.NUM_DRONES):
            with open(csv_dir + "/pos" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 0:3, :]])), delimiter=",")
            ####
            with open(csv_dir + "/rpy" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 6:9, :]])), delimiter=",")
            ####
            with open(csv_dir + "/vel" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 3:6, :]])), delimiter=",")
            ####
            with open(csv_dir + "/ang_vel" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 9:12, :]])), delimiter=",")
            ####
            with open(csv_dir + "/rpm" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 12:16, :]])), delimiter=",")
            ####
            with open(csv_dir + "/pwm" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, (self.states[i, 12:16, :] - 4070.3) / 0.2685])),
                           delimiter=",")
            with open(csv_dir + "/psi" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.ref[i, 6, :]])), delimiter=",")
            with open(csv_dir + "/eta" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.ref[i, 7:9, :]])), delimiter=",")
            with open(csv_dir + "/mu" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.ref[i, 9:11, :]])), delimiter=",")
            with open(csv_dir + "/pos_ref" + str(i) + ".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.ref[i, 0:3, :]])), delimiter=",")

    ################################################################################

    def plot_ff(self, block=True, fig=None, axs=None):
        if fig is None:
            fig, axs = plt.subplots(4, 2)
        t = self.timestamps[0, :]
        axs[0, 0].plot(t, self.states[:, 0, :].T)
        axs[1, 0].plot(t, self.states[:, 2, :].T)
        axs[2, 0].plot(self.states[:, 0, :].T, self.states[:, 2, :].T)

        axs[0, 1].plot(t, self.states[:, 7, :].T)
        axs[1, 1].plot(t, self.states[:, 10, :].T)
        axs[2, 1].plot(t, self.controls[:, 0, :].T)
        axs[3, 1].plot(t, self.controls[:, 2, :].T)
        for i in range(4):
            for j in range(2):
                axs[i, j].grid(True)
            # axs[i, j].legend(loc='upper right',
            #                  frameon=True
            #                  )
        fig.subplots_adjust(left=0.15,
                            bottom=0.05,
                            right=0.99,
                            top=0.98,
                            wspace=0.5,
                            hspace=0.5
                            )
        plt.show(block=block)

    ################################################################################

    def plot(self, block=True, fig=None, axs=None):
        if fig is None:
            fig, axs = plt.subplots(6, 2)
        t = self.timestamps[0, :]

        #### Column ################################################
        col = 0

        #### XYZ ###################################################
        row = 0
        # axs[row, col].plot(t, self.states[:, 0, :].T - self.states[:, 3, :].T)
        axs[row, col].plot(t, self.states[:, 4, :].T)
        # axs[row].plot(t, )
        # axs[row].plot(t, )
        axs[row, col].set_xlabel('time (s)')
        # axs[row].set_ylabel('$e_r$ (m)')
        # axs[row].legend(('$e_x$', '$e_y$', '$e_z$'))
        # axs[row].set_ylim(-0.2, 0.12)
        # axs[row].set_yticks((-0.2, -0.1, 0, 0.1))

        row = 1
        axs[row, col].plot(t, self.states[:, 2, :].T - self.states[:, 5, :].T)
        # axs[row].plot(t, )
        # axs[row].plot(t, )
        axs[row, col].set_xlabel('time (s)')
        # axs[row].set_ylabel('$e_r$ (m)')
        # axs[row].legend(('$e_x$', '$e_y$', '$e_z$'))
        # axs[row].set_ylim(-0.2, 0.12)
        # axs[row].set_yticks((-0.2, -0.1, 0, 0.1))

        row = 2
        # axs[row].plot(t, self.states[:, 7, :].T - self.states[:, 10, :].T)
        axs[row, col].plot(t, self.states[:, 22, :].T)
        # axs[row].plot(t, )
        axs[row, col].set_xlabel('time (s)')
        # axs[row].set_ylabel('$e_y$ (m)')
        # axs[row, col].set_ylim(-0.2, 0.08)
        #
        row = 3
        # axs[row].plot(self.states[:, 0, :].T, self.states[:, 2, :].T)
        axs[row, col].plot(self.states[:, 3, :].T, self.states[:, 5, :].T)
        axs[row, col].set_xlabel('x (m)')
        axs[row, col].set_ylabel('z(m)')
        # axs[row, col].set_ylim(-0.05, 0.1)

        row = 4
        axs[row, col].plot(t, self.controls[:, 0, :].T)

        col = 1

        #### Controls ###############################################
        row = 0
        axs[row, col].plot(t, self.controls[:, 1, :].T)

        row = 1
        axs[row, col].plot(t, self.controls[:, 2, :].T)

        row = 2
        axs[row, col].plot(t, self.gp_terms[:, 0, :].T)

        row = 3
        axs[row, col].plot(t, self.gp_terms[:, 1, :].T)

        row = 4
        axs[row, col].plot(t, self.gp_terms[:, 2, :].T)

        row = 5
        axs[row, col].plot(t, self.gp_terms[:, 3, :].T)


        # axs[row].set_xlabel('time (s)')
        # axs[row].set_ylabel('F (N)')
        # axs[row].set_ylim(40, 46.1)
        # axs[row].set_yticks((40, 42, 44, 46))
        #
        # row = 2
        # axs[row].plot(t, self.controls[:, 1, :].T)
        # axs[row].plot(t, self.controls[:, 2, :].T)
        # axs[row].plot(t, self.controls[:, 3, :].T)
        # axs[row].set_xlabel('time (s)')
        # axs[row].set_ylabel(r'$\tau$ (Nm)')
        # axs[row].legend((r'$\tau_x$', r'$\tau_y$', r'$\tau_z$'))
        # axs[row].set_ylim(-1.1, 1.5)
        # axs[row].set_yticks((-1, -0.5, 0, 0.5, 1))

        # row = 2
        # axs[row, col].plot(t, self.controls[:, 2, :].T)
        # axs[row, col].set_xlabel('time (s)')
        # axs[row, col].set_ylabel(r'$\tau_y$ (Nm)')
        #
        # row = 3
        # axs[row, col].plot(t, self.controls[:, 3, :].T)
        # axs[row, col].set_xlabel('time (s)')
        # axs[row, col].set_ylabel(r'$\tau_z$ (Nm)')

        #### Drawing options #######################################
        for i in range(6):
            for j in range(2):
                axs[i, j].grid(True)
            # axs[i, j].legend(loc='upper right',
            #                  frameon=True
            #                  )
        fig.subplots_adjust(left=0.15,
                            bottom=0.05,
                            right=0.99,
                            top=0.98,
                            wspace=0.5,
                            hspace=0.5
                            )
        plt.show(block=block)

    def plot3D(self):
        x = self.states[0, 9, :].tolist()
        y = self.states[0, 10, :].tolist()
        z = self.states[0, 11, :].tolist()
        xr = self.states[0, 6, :].tolist()
        yr = self.states[0, 7, :].tolist()
        zr = self.states[0, 8, :].tolist()
        points = np.array([xr, yr, zr]).T.reshape(-1, 1, 3)
        # segments = np.concatenate([points[:-1], points[1:]], axis=1)
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        # Create a continuous norm to map from data points to colors

        # idx = {}
        # # idx['A'] = 0
        # idx['B'] = (np.linalg.norm(points[:, 0, 0:2] - np.array([-0.3, 0]), axis=1)).argmin()
        # idx['C'] = (np.linalg.norm(points[:, 0, 0:2], axis=1)).argmin()
        # idx['D'] = (np.linalg.norm(points[:, 0, :] - np.array([0.3, 0, 1.1]), axis=1)).argmin()
        # for (k, v) in idx.items():
        #     ax.scatter(xr[v], yr[v], zr[v], marker='x', color='black')
        #     ax.text(xr[v], yr[v], zr[v] + 0.1, k)

        # x, y, z, xr, yr, zr = x[idx['B']:], y[idx['B']:], z[idx['B']:], xr[idx['B']:], yr[idx['B']:], zr[idx['B']:]
        ax.plot3D(x, y, z, 'red')
        ax.plot3D(xr, yr, zr, 'blue')
        ax.legend(('Simulation', 'Reference'))

        # traj_break_idx = np.argmax(np.abs(y) < 1e-4)
        # ax.scatter(x[traj_break_idx], y[traj_break_idx], z[traj_break_idx])
        # ax.scatter(0, 0, 0, marker='*')
        # ax.text(x[0], y[0], z[0], str([x[0], y[0], z[0]]))
        # ax.text(x[-1], y[-1], z[-1], "[{:.1f}, {:.1f}, {:.1f}]".format(x[-1], y[-1], z[-1]))
        ax.set_xlim(min(x) - 0.3, max(x) + 0.3)
        ax.set_ylim(min(y) - 0.3, max(y) + 0.3)
        ax.set_zlim(min(z) - 0.1, max(z) + 0.3)
        ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        plt.show(block=True)

    def plot2D(self):
        fig, axs = plt.subplots(2, 2)
        t = self.timestamps[0, :]

        #### Column ################################################
        col = 0

        #### XYZ ###################################################
        row = 0
        axs[row, col].plot(t, self.states[:, 0, :].T)
        axs[row, col].plot(t, self.states[:, 1, :].T)
        axs[row, col].set_xlabel('time (s)')
        axs[row, col].set_ylabel('position (m)')
        axs[row, col].legend(('$x_L$', '$z_L$'))
        # axs[row, col].set_ylim(-0.2, 0.05)

        row = 1
        axs[row, col].plot(t, self.states[:, 2, :].T)
        axs[row, col].plot(t, self.states[:, 3, :].T)
        axs[row, col].set_xlabel('time (s)')
        axs[row, col].set_ylabel('angles (rad)')
        axs[row, col].legend((r'$\theta$', r'$\alpha$'))
        # axs[row, col].set_yticks([-0.25, 0, 0.25, 0.5])
        # axs[row, col].set_ylim(-0.2, 0.08)

        col = 1

        #### Controls ###############################################
        row = 0
        axs[row, col].plot(t, self.controls[:, 0, :].T)
        axs[row, col].set_xlabel('time (s)')
        axs[row, col].set_ylabel('F (N)')

        row = 1
        axs[row, col].plot(t, self.controls[:, 2, :].T)
        axs[row, col].set_xlabel('time (s)')
        axs[row, col].set_ylabel(r'$\tau$ (Nm)')

        #### Drawing options #######################################
        for i in range(2):
            for j in range(2):
                axs[i, j].grid(True)
                # axs[i, j].legend(loc='upper right',
                #                  frameon=True
                #                  )
        fig.subplots_adjust(left=0.15,
                            bottom=0.05,
                            right=0.99,
                            top=0.98,
                            wspace=0.5,
                            hspace=0.5
                            )
        plt.show(block=False)

    def save_log(self, filename):
        simu = dict(xref=self.states[0, 0, :],
                    yref=self.states[0, 1, :],
                    zref=self.states[0, 2, :],
                    stateEstimate_x=self.states[0, 3, :],
                    stateEstimate_y=self.states[0, 4, :],
                    stateEstimate_z=self.states[0, 5, :],
                    ctrlGeom_eRx=self.states[0, 23, :],
                    ctrlGeom_eRy=self.states[0, 24, :],
                    ctrlGeom_eRz=self.states[0, 25, :],
                    ctrlGeom_psi=self.states[0, 22, :],
                    ctrlGeom_thrust=self.controls[0, 0, :],
                    ctrlGeom_Mx=self.controls[0, 1, :],
                    ctrlGeom_My=self.controls[0, 2, :],
                    ctrlGeom_Mz=self.controls[0, 3, :],
                    ctrlGeom_eta0=self.gp_terms[0, 0, :],
                    ctrlGeom_eta1=self.gp_terms[0, 1, :],
                    ctrlGeom_mu0=self.gp_terms[0, 2, :],
                    ctrlGeom_mu1=self.gp_terms[0, 3, :]
                    )
        header = simu.keys()
        with open('../files/'+filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(len(simu['stateEstimate_x'])):
                row = [simu[key][i] for key in simu.keys()]
                writer.writerow(row)

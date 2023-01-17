import numpy as np
from ctrl.GPGeomControl import GPGeomControl
from ctrl.RobustGeomControl import RobustGeomControl
import torch


class RobustGPGeomControl(GPGeomControl, RobustGeomControl):
    def __init__(self,
                 model,
                 data,
                 drone_type='cf2',
                 train_x=torch.tensor([0, 0, 0, 0]).float(),
                 train_y=torch.tensor([0]).float()
                 ):
        super().__init__(model, data, drone_type, train_x, train_y)

    def compute_delta(self):
        pred = self.eval_model(torch.linspace(0, 1, 500).float())
        self.delta_R = 0.001 * max(np.hstack((np.sqrt(pred[0].variance.detach().numpy()), np.sqrt(pred[1].variance.detach().numpy()))))

    def _mu_R(self, cur_quat, cur_ang_vel, rot_e, ang_vel_e):
        # GP
        q = cur_quat[0:2]  # TODO
        w = cur_ang_vel[0:2]
        # pred = self.eval_model(torch.tensor([np.hstack((q, w))]).float())
        pred = self.eval_model(torch.tensor([q[1]]).float())
        # self.delta_R = 2*max((np.sqrt(pred[0].variance.numpy())/200, np.sqrt(pred[1].variance.numpy())/200))
        mu_R = RobustGeomControl._mu_R(self, cur_quat, cur_ang_vel, rot_e, ang_vel_e)
        return np.array([float(-pred[0].mean/200), float(-pred[1].mean/200), 0]),  mu_R

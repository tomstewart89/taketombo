"""Dynamic model of taketombo

author: Christof Dubs
"""
import numpy as np
from numpy import sin, cos, tan
from scipy.integrate import odeint
from pyrotation import Quaternion, quat_from_angle_vector


class ModelParam:
    """Physical parameters of taketombo

    Indices 1, 2 and 3 generally refer to the following rigid bodies:
        1: upper body (with propellers attached)
        2: middle body
        3: lower body


    Attributes:
        c_d: propeller drag per thrust constant [Nm/N]
        c_t: propeller thrust per squared angular velocity constant [N/rad^2]
        g: Gravitational constant [m/s^2]
        l1 : distance from center of mass of upper body to first motor's axis [m]
        l2 : distance from first motor's axis to center of mass of middle body [m]
        l3 : distance from center of mass of middle body to second motor's axis [m]
        l4 : distance from second motor's axis to center of mass of lower body [m]
        lp : distance from center of mass of upper body to propeller [m]
        m1: Mass of upper body [kg]
        m2: Mass of middle body [kg]
        m3: Mass of lower body [kg]
        mp: Mass of propeller [kg]
        tau: time constant of speed controlled motor [s]
        J1_xx / J1_yy / J1_zz: Mass moment of inertia of upper body wrt. its center of mass around x / y / z axis [kg*m^2]
        J2_xx / J2_yy / J2_zz: Mass moment of inertia of middle body wrt. its center of mass around x / y / z axis [kg*m^2]
        J3_xx / J3_yy / J3_zz: Mass moment of inertia of lower body wrt. its center of mass around x / y / z axis [kg*m^2]
        Jp_xx / Jp_yy / Jp_zz: Mass moment of inertia of propeller wrt. its center of mass around x / y / z axis [kg*m^2]
    """

    def __init__(self,):
        """Initializes the parameters to default values"""
        self.c_d = 0.020
        self.c_t = 1e-4  # 1N / (100rad/s)^2
        self.g = 9.81
        self.l1 = 0.025
        self.l2 = 0.025
        self.l3 = 0.050
        self.l4 = 0.050
        self.lp = 0.025
        self.m1 = 0.100
        self.m2 = 0.100
        self.m3 = 0.500
        self.mp = 0.050
        self.tau = 0.100
        self.J1_xx = 1e-4
        self.J1_yy = 1e-4
        self.J1_zz = 1e-4
        self.J2_xx = 1e-4
        self.J2_yy = 1e-4
        self.J2_zz = 1e-4
        self.J3_xx = 1e-3
        self.J3_yy = 1e-3
        self.J3_zz = 1e-3
        self.Jp_xx = 1e-9
        self.Jp_yy = 1e-7
        self.Jp_zz = 1e-7

    def is_valid(self,):
        """Checks validity of parameter configuration

        Returns:
            bool: True if valid, False if invalid.
        """
        return self.c_d >= 0 and self.c_t > 0 and self.g > 0 and self.l1 > 0 and self.l2 > 0 and self.l3 > 0 and self.l4 > 0 and self.lp > 0 and self.m1 > 0 and self.m2 > 0 and self.m3 > 0 and self.mp > 0 and self.tau > 0 and self.J1_xx > 0 and self.J1_yy > 0 and self.J1_zz > 0 and self.J2_xx > 0 and self.J2_yy > 0 and self.J2_zz > 0 and self.J3_xx > 0 and self.J3_yy > 0 and self.J3_zz > 0 and self.Jp_xx > 0 and self.Jp_yy > 0 and self.Jp_zz > 0


# state size and indices
X_IDX = 0
Y_IDX = 1
Z_IDX = 2
ALPHA_X_IDX = 3
ALPHA_Y_IDX = 4
BETA_1_IDX = 5
BETA_2_IDX = 6
Q_1_W_IDX = 7
Q_1_X_IDX = 8
Q_1_Y_IDX = 9
Q_1_Z_IDX = 10

U_IDX = 11
V_IDX = 12
W_IDX = 13
ALPHA_X_DOT_IDX = 14
ALPHA_Y_DOT_IDX = 15
BETA_1_DOT_IDX = 16
BETA_2_DOT_IDX = 17
PHI_DOT_IDX = 18
THETA_DOT_IDX = 19
PSI_DOT_IDX = 20

STATE_SIZE = 21


class ModelState:
    """Class to represent state of taketombo

    The state is stored in this class as a numpy array, for ease of interfacing
    with scipy. Numerous properties / setters allow interfacing with this class
    without knowledge about the state index definitions.

    Attributes:
        x (numpy.ndarray): array representing the full state
    """

    def __init__(self, x0=None, skip_checks=False):
        """Initializes attributes

        args:
            x0 (np.ndarray, optional): initial state. Set to default values if not specified
            skip_checks (bool, optional): if set to true and x0 is provided, x0 is set without checking it.
        """
        if skip_checks and x0 is not None:
            self.x = x0
            return

        if x0 is None or not self.set_state(x0):
            self.x = np.zeros(STATE_SIZE, dtype=np.float)
            self.x[Q_1_W_IDX] = 1

    def normalize_quaternions(self):
        """Normalize the rotation quaternions"""
        self.q1 *= 1.0 / np.linalg.norm(self.q1)

    def set_state(self, x0):
        """Set the state.

        This function allows to set the initial state.

        args:
            x0 (numpy.ndarray): initial state

        Returns:
            bool: True if state could be set successfully, False otherwise.
        """
        if not isinstance(x0, np.ndarray):
            print(
                'called set_state with argument of type {} instead of numpy.ndarray. Ignoring.'.format(
                    type(x0)))
            return False

        # make 1D version of x0
        x0_flat = x0.flatten()
        if len(x0_flat) != STATE_SIZE:
            print(
                'called set_state with array of length {} instead of {}. Ignoring.'.format(
                    len(x0_flat), STATE_SIZE))
            return False

        q1_norm = np.linalg.norm(x0_flat[Q_1_W_IDX:Q_1_Z_IDX + 1])

        # quaternion check
        if q1_norm == 0:
            return false

        self.x = x0_flat
        self.normalize_quaternions()
        return True

    @property
    def q1(self):
        return self.x[Q_1_W_IDX:Q_1_Z_IDX + 1]

    @q1.setter
    def q1(self, value):
        if isinstance(value, Quaternion):
            self.x[Q_1_W_IDX:Q_1_Z_IDX + 1] = value.q
            return
        if isinstance(value, np.ndarray):
            self.x[Q_1_W_IDX:Q_1_Z_IDX + 1] = value
            return
        print('failed to set x')

    @property
    def q2(self):
        return Quaternion(self.q1) * quat_from_angle_vector(np.array([self.x[ALPHA_X_IDX], 0, 0]))

    @property
    def q3(self):
        return self.q2 * quat_from_angle_vector(np.array([0, self.x[ALPHA_Y_IDX], 0]))

    @property
    def roll_pitch_yaw(self):
        return Quaternion(self.q1).get_roll_pitch_yaw()

    @property
    def roll_pitch_yaw_rate(self):
        return self.x[PHI_DOT_IDX:PSI_DOT_IDX + 1]

    @roll_pitch_yaw_rate.setter
    def roll_pitch_yaw_rate(self, value):
        self.x[PHI_DOT_IDX:PSI_DOT_IDX + 1] = value

    @property
    def pos(self):
        return self.x[X_IDX:Z_IDX + 1]

    @pos.setter
    def pos(self, value):
        self.x[X_IDX:Z_IDX + 1] = value

    @property
    def vel(self):
        return self.x[U_IDX:PSI_DOT_IDX + 1]

    @vel.setter
    def vel(self, value):
        self.x[U_IDX:PSI_DOT_IDX + 1] = value

    @property
    def B_vel(self):
        return self.x[U_IDX:W_IDX + 1]

    @B_vel.setter
    def B_vel(self, value):
        self.x[U_IDX:W_IDX + 1] = value

    @property
    def alpha(self):
        return self.x[ALPHA_X_IDX:ALPHA_Y_IDX + 1]

    @alpha.setter
    def alpha(self, value):
        self.x[ALPHA_X_IDX:ALPHA_Y_IDX + 1] = value

    @property
    def alpha_dot(self):
        return self.x[ALPHA_X_DOT_IDX:ALPHA_Y_DOT_IDX + 1]

    @alpha_dot.setter
    def alpha_dot(self, value):
        self.x[ALPHA_X_DOT_IDX:ALPHA_Y_DOT_IDX + 1] = value

    @property
    def beta(self):
        return self.x[BETA_1_IDX:BETA_2_IDX + 1]

    @beta.setter
    def beta(self, value):
        self.x[BETA_1_IDX:BETA_2_IDX + 1] = value

    @property
    def beta_dot(self):
        return self.x[BETA_1_DOT_IDX:BETA_2_DOT_IDX + 1]

    @beta_dot.setter
    def beta_dot(self, value):
        self.x[BETA_1_DOT_IDX:BETA_2_DOT_IDX + 1] = value


class DynamicModel:
    """Simulation interface for the taketombo

    Attributes:
        p (ModelParam): physical parameters
        state (ModelState): 21-dimensional state

    Functions that are not meant to be called from outside the class (private methods) are prefixed with a single underline.
    """

    def __init__(self, param, x0=None):
        """Initializes attributes to default values

        args:
            param (ModelParam): parameters of type ModelParam
            x0 (ModelState, optional): initial state. Set to default state if not specified
        """
        self.p = param
        if not param.is_valid():
            print('Warning: not all parameters set!')

        if x0 is not None:
            if not isinstance(x0, ModelState):
                print('invalid type passed as initial state')
                self.state = ModelState()
            else:
                self.state = x0
        else:
            self.state = ModelState()

    def simulate_step(self, delta_t, motor_cmds):
        """Simulate one time step

        Simulates the changes of the state over a time interval.

        args:
            delta_t: time step [s]
            motor_cmds (np.ndarray): motor commands: [servo x-angle [rad], servo y-angle [rad], counter-clockwise propeller-motor speed [rad/s], clockwise propeller-motor speed [rad/s]]
        """
        t = np.array([0, delta_t])
        self.state.x = odeint(self._x_dot, self.state.x, t, args=(motor_cmds,))[-1]

        # normalize quaternions
        self.state.normalize_quaternions()

    def get_visualization(self, state=None):
        """Get visualization of the system for plotting

        Usage example:
            v = model.get_visualization()
            plt.plot(*v['upper_body'])

        args:
            state (ModelState, optional): state. If not specified, the internal state is checked

        Returns:
            dict: dictionary with keys "upper_body", "middle_body" and "lower_body". The value for each key is a list with three elements: a list of x coordinates, a list of y coordinates and a list of z coordinates.
        """
        if state is None:
            state = self.state

        vis = {}

        r_OSi = self._compute_r_OSi(state)
        vis['upper_body'] = self._compute_body_visualization(
            r_OSi[0], self.p.lp, self.p.lp + self.p.l1, Quaternion(state.q1))
        vis['middle_body'] = self._compute_body_visualization(
            r_OSi[1], self.p.l2, self.p.l3, state.q2)
        vis['lower_body'] = self._compute_body_visualization(
            r_OSi[2], self.p.l4, 2 * self.p.l4, state.q3)
        return vis

    def _x_dot(self, x, t, motor_cmds):
        """computes the derivative of the state

        This function returns an numpy.array of the derivatives of the states, given the current state and inputs.

        Its signature is compatible with scipy.integrate.odeint's first callable argument.

        args:
            x (numpy.ndarray): state at which the state derivative function is evaluated
            t: time [s]. Since this system is time invariant, this argument is unused.
            motor_cmds (np.ndarray): motor commands: [servo x-angle [rad], servo y-angle [rad], counter-clockwise propeller-motor speed [rad/s], clockwise propeller-motor speed [rad/s]]
        returns:
            ModelState containing the time derivatives of all states
        """
        eval_state = ModelState(x, skip_checks=True)

        xdot = ModelState()

        xdot.vel = self._compute_vel_dot(eval_state, motor_cmds)

        omega_1 = self._get_upper_body_omega(eval_state)

        q_IB = Quaternion(eval_state.q1)

        xdot.q1 = q_IB.q_dot(omega_1, frame='body')

        R_IB = q_IB.rotation_matrix()

        xdot.pos = np.dot(R_IB, eval_state.B_vel)
        xdot.alpha = eval_state.alpha_dot
        xdot.beta = eval_state.beta_dot

        return xdot.x

    def _get_upper_body_omega(self, state):
        """computes the angular velocity (x/y/z) of the upper body

        args:
            state (ModelState): current state
        returns:
            array containing angular velocity of upper body [rad/s]
        """
        [phi, theta, psi] = state.roll_pitch_yaw
        [phi_dot, theta_dot, psi_dot] = state.roll_pitch_yaw_rate

        B1_omega_IB1 = np.zeros(3)

        x0 = cos(phi)
        x1 = sin(phi)
        x2 = psi_dot * cos(theta)
        B1_omega_IB1[0] = phi_dot - psi_dot * sin(theta)
        B1_omega_IB1[1] = theta_dot * x0 + x1 * x2
        B1_omega_IB1[2] = -theta_dot * x1 + x0 * x2

        return B1_omega_IB1

    def _compute_vel_dot(self, state, motor_cmds):
        """computes derivative of velocity states

        args:
            state (ModelState): current state
            motor_cmds (np.ndarray): motor commands: [servo x-angle [rad], servo y-angle [rad], counter-clockwise propeller-motor speed [rad/s], clockwise propeller-motor speed [rad/s]]

        Returns: array containing the time derivative of the velocity states
        """
        [phi, theta, psi] = state.roll_pitch_yaw
        [phi_dot, theta_dot, psi_dot] = state.roll_pitch_yaw_rate
        [alphax, alphay] = state.alpha
        [beta1, beta2] = state.beta
        [alphax_dot, alphay_dot] = state.alpha_dot
        [beta1_dot, beta2_dot] = state.beta_dot
        [u, v, w] = state.B_vel
        g = self.p.g

        [alphax_cmd, alphay_cmd, omega_1_cmd, omega_2_cmd] = motor_cmds

        if omega_1_cmd < 0:
            print('omega_1_cmd = {} is not >= 0 as it should be. The model only works properly when spinning motor 1 in positive direction'.format(
                omega_1_cmd))

        if omega_2_cmd > 0:
            print('omega_2_cmd = {} is not <= 0  as it should be. The model only works properly when spinning motor 2 in negative direction'.format(
                omega_2_cmd))

        # simulate servo position control (critically damped)
        omega_x_cmd = 1 / (4 * self.p.tau) * (alphax_cmd - alphax)
        omega_y_cmd = 1 / (4 * self.p.tau) * (alphay_cmd - alphay)

        A = np.zeros([10, 10])
        b = np.zeros(10)

        # auto-generated symbolic expressions
        x0 = 2 * self.p.mp
        x1 = self.p.m1 + self.p.m2 + self.p.m3 + x0
        x2 = cos(alphay)
        x3 = self.p.l4 * x2
        x4 = sin(alphax)
        x5 = x4**2
        x6 = cos(alphax)
        x7 = self.p.m3 * (-x3 * x5 - x3 * x6**2)
        x8 = cos(phi)
        x9 = self.p.lp * x0
        x10 = -self.p.l1 * x8
        x11 = sin(phi)
        x12 = x11 * x4
        x13 = self.p.l2 * x6
        x14 = self.p.l2 * x12 + x10 - x13 * x8
        x15 = self.p.m2 * x14
        x16 = self.p.l3 * x4
        x17 = self.p.l3 * x6
        x18 = x3 * x4
        x19 = x6 * x8
        x20 = -x12 + x19
        x21 = x4 * x8
        x22 = x11 * x6
        x23 = -x21 - x22
        x24 = x20 * x4 + x23 * x6
        x25 = x3 * x6
        x26 = x20 * x6 - x23 * x4
        x27 = x10 + x11 * x16 - x17 * x8 - x18 * x24 - x25 * x26
        x28 = self.p.m3 * x27
        x29 = x15 + x28 + x8 * x9
        x30 = cos(theta)
        x31 = x11 * x30
        x32 = self.p.l1 * x30
        x33 = -x11 * x32
        x34 = x21 * x30
        x35 = -self.p.l2 * x34 - x13 * x31 + x33
        x36 = x30 * x8
        x37 = x22 * x30 + x34
        x38 = -x12 * x30 + x19 * x30
        x39 = x37 * x4 + x38 * x6
        x40 = x37 * x6 - x38 * x4
        x41 = -x16 * x36 - x17 * x31 - x18 * x39 - x25 * x40 + x33
        x42 = self.p.m2 * x35 + self.p.m3 * x41 + x31 * x9
        x43 = self.p.m2 * x13
        x44 = x17 + x25
        x45 = self.p.m3 * x44
        x46 = sin(alphay)
        x47 = self.p.l4 * self.p.m3 * x46
        x48 = x4 * x47
        x49 = self.p.l1 + x13
        x50 = self.p.m2 * x49
        x51 = self.p.l1 + x44
        x52 = self.p.m3 * x51
        x53 = x50 + x52 - x9
        x54 = x24 * x47
        x55 = -x54
        x56 = sin(theta)
        x57 = self.p.l1 * x56
        x58 = -x57
        x59 = -x13 * x56 + x58
        x60 = self.p.l4 * x2 * x56
        x61 = self.p.l4 * x46
        x62 = -x17 * x56 - x39 * x61 + x58 - x6 * x60
        x63 = self.p.m2 * x59 + self.p.m3 * x62 + x56 * x9
        x64 = self.p.l2 * self.p.m2 * x4
        x65 = x16 + x18
        x66 = self.p.m3 * x65
        x67 = x64 + x66
        x68 = x47 * x6
        x69 = x26 * x47
        x70 = -x16 * x56 - x4 * x60 + x40 * x61
        x71 = self.p.m3 * x70 - x56 * x64
        x72 = x2**2
        x73 = x46**2
        x74 = self.p.l2**2 * self.p.m2 * x5
        x75 = self.p.J2_xx + self.p.J3_xx * x72 + self.p.J3_zz * x73 + self.p.m3 * x65**2 + x74
        x76 = cos(beta1)
        x77 = x76**2
        x78 = cos(beta2)
        x79 = x78**2
        x80 = sin(beta1)
        x81 = x80**2
        x82 = sin(beta2)
        x83 = x82**2
        x84 = self.p.lp**2 * x0
        x85 = self.p.Jp_xx * x8
        x86 = x76 * x80
        x87 = x78 * x82
        x88 = self.p.Jp_yy * x8
        x89 = x2 * x23 * x46
        x90 = -self.p.J3_xx * x89 + self.p.J3_zz * x89 + x65 * x69
        x91 = -x51 * x54 + x85 * x86 + x85 * x87 - x86 * x88 - x87 * x88 + x90
        x92 = x2 * x38 - x46 * x56
        x93 = self.p.J3_zz * x92
        x94 = -x2 * x56 - x38 * x46
        x95 = self.p.J3_xx * x94
        x96 = -self.p.J2_xx * x56 + x2 * x95 + x46 * x93 - x56 * x74 + x66 * x70
        x97 = x31 * x80 - x56 * x76
        x98 = x31 * x82 - x56 * x78
        x99 = x31 * x76 + x56 * x80
        x100 = x31 * x78 + x56 * x82
        x101 = -self.p.J1_xx * x56 + self.p.Jp_xx * x76 * x97 + self.p.Jp_xx * x78 * x98 - \
            self.p.Jp_yy * x100 * x82 - self.p.Jp_yy * x80 * x99 + x50 * x59 + x52 * x62 - x56 * x84 + x96
        x102 = self.p.J3_yy * x20
        x103 = self.p.l4**2 * self.p.m3 * x73
        x104 = -self.p.Jp_zz * x11
        x105 = x8**2
        x106 = x11**2
        x107 = 2 * self.p.Jp_zz
        x108 = self.p.Jp_xx * x105
        x109 = self.p.Jp_yy * x105
        x110 = x20**2
        x111 = x23**2
        x112 = x11 * x30 * x8
        x113 = x2 * x23
        x114 = x23 * x46
        x115 = self.p.J1_yy * x112 - self.p.J1_zz * x112 + self.p.J2_yy * x20 * x37 + self.p.J2_zz * x23 * x38 - 2 * self.p.Jp_zz * x11 * x36 + x100 * x78 * x88 + \
            x102 * x37 + x112 * x84 + x113 * x93 - x114 * x95 + x15 * x35 + x28 * x41 - x54 * x62 + x69 * x70 + x76 * x88 * x99 + x80 * x85 * x97 + x82 * x85 * x98
        x116 = self.p.Jp_zz * x36
        x117 = x56**2
        x118 = x30**2
        x119 = x106 * x118
        x120 = x105 * x118
        x121 = x37**2
        x122 = g * x56
        x123 = -self.p.m2 * x122
        x124 = psi_dot * x56
        x125 = theta_dot * x11
        x126 = x124 * x125
        x127 = theta_dot * x8
        x128 = psi_dot * x30
        x129 = x11 * x128
        x130 = x127 + x129
        x131 = self.p.m1 * x130
        x132 = w * x130
        x133 = psi_dot * x36 - x125
        x134 = self.p.m1 * x133
        x135 = phi_dot * x133
        x136 = self.p.l2 * x4
        x137 = phi_dot - x124
        x138 = alphax_dot + x137
        x139 = self.p.m2 * (w + x136 * x138)
        x140 = x130 * x139
        x141 = -self.p.lp * x137 + v
        x142 = 2 * self.p.mp * x141
        x143 = self.p.m2 * theta_dot
        x144 = psi_dot * x11 * x57
        x145 = x124 * x21
        x146 = psi_dot * x11 * x56
        x147 = x143 * (self.p.l2 * x145 + x13 * x146 + x144)
        x148 = -x13 * x133
        x149 = alphax_dot * self.p.m2 * (x130 * x136 + x148)
        x150 = self.p.l1 * x137
        x151 = self.p.m2 * (v + x13 * x138 + x150)
        x152 = -x133 * x151
        x153 = -self.p.l1 * x133
        x154 = -x127 - x129
        x155 = phi_dot * self.p.m2 * (-x136 * x154 + x148 + x153)
        x156 = x138 * x16
        x157 = x138 * x18
        x158 = x130 * x6
        x159 = x133 * x4
        x160 = x158 + x159
        x161 = alphay_dot + x160
        x162 = x161 * x6
        x163 = x133 * x6
        x164 = -x130 * x4 + x163
        x165 = x162 - x164 * x4
        x166 = x165 * x61
        x167 = self.p.m3 * (w + x156 + x157 + x166)
        x168 = x138 * x17 + x138 * x25
        x169 = x161 * x4
        x170 = x164 * x6 + x169
        x171 = x170 * x61
        x172 = self.p.m3 * (v + x150 + x168 - x171)
        x173 = self.p.m3 * theta_dot
        x174 = x12 * x124 - x124 * x19
        x175 = -x124 * x22 - x145
        x176 = x174 * x6 + x175 * x4
        x177 = -x174 * x4 + x175 * x6
        x178 = alphay_dot * self.p.m3
        x179 = -x133 * x17
        x180 = x154 * x4 + x163
        x181 = -x159
        x182 = x154 * x6 + x181
        x183 = x180 * x4 + x182 * x6
        x184 = x180 * x6 - x182 * x4
        x185 = alphax_dot * self.p.m3
        x186 = -x158 + x181
        x187 = x162 + x186 * x6
        x188 = -x169 - x186 * x4
        x189 = phi_dot * self.p.m3 * (
            x153 - x154 * x16 + x179 - x18 * x183 - x184 * x25) - self.p.m3 * x122 + x130 * x167 - x133 * x172 + x173 * (
            self.p.l3 * x124 * x4 * x8 + x144 + x146 * x17 - x176 * x18 - x177 * x25) + x178 * (
            x166 * x6 + x171 * x4) + x185 * (
                x130 * x16 + x165 * x18 - x170 * x25 + x179 - x18 * x187 - x188 * x25)
        x190 = g * self.p.m1
        x191 = g * self.p.m2
        x192 = x191 * x31
        x193 = g * x11 * x30
        x194 = psi_dot * theta_dot * x30
        x195 = self.p.m1 * x137
        x196 = w * x137
        x197 = alphax_dot * x138
        x198 = -x197 * x64
        x199 = -psi_dot * x32
        x200 = x143 * (-x128 * x13 + x199)
        x201 = -x137 * x139
        x202 = self.p.lp * x130 + u
        x203 = 2 * self.p.mp * x202
        x204 = -self.p.l1 * x130 + u
        x205 = self.p.m2 * (-x13 * x130 - x133 * x136 + x204)
        x206 = x133 * x205
        x207 = g * self.p.m3
        x208 = phi_dot * self.p.l4 * self.p.m3 * x46
        x209 = self.p.l4 * x138 * x46
        x210 = self.p.m3 * (-x130 * x17 - x133 * x16 - x165 * x25 - x170 * x18 + x204)
        x211 = x133 * x210 - x137 * x167 + x173 * (-x128 * x17 - x128 * x25 - x176 * x61 + x199) + x178 * (
            -x170 * x3 - x209 * x6) - x183 * x208 + x185 * (-x156 - x157 - x187 * x61) + x207 * x31
        x212 = beta1_dot**2 * self.p.c_t
        x213 = beta2_dot**2 * self.p.c_t
        x214 = x191 * x36
        x215 = -x194 * x64
        x216 = x197 * x43
        x217 = x137 * x151
        x218 = -x130 * x205
        x219 = -x130 * x210 + x137 * x172 + x173 * (-x128 * x16 - x128 * x18 + x177 * x61) + x178 * (
            x165 * x3 - x209 * x4) + x184 * x208 + x185 * (x168 + x188 * x61) + x207 * x36
        x220 = 1 / self.p.tau
        x221 = x160 * x164
        x222 = -self.p.J2_xx * x194 - self.p.J2_yy * x221 + self.p.J2_zz * x221
        x223 = -self.p.J1_xx * x194
        x224 = x130 * x133
        x225 = self.p.J1_zz * x224
        x226 = -self.p.J1_yy * x224
        x227 = self.p.lp * self.p.mp
        x228 = self.p.mp * x133
        x229 = 2 * self.p.lp * (self.p.mp * x193 - self.p.mp * x196 + x194 * x227 + x202 * x228)
        x230 = x136 * (x214 + x215 + x216 + x217 + x218)
        x231 = x192 + x198 + x200 + x201 + x206
        x232 = beta1_dot + x133
        x233 = x137 * x76
        x234 = x130 * x80
        x235 = x233 + x234
        x236 = x232 * x235
        x237 = self.p.Jp_xx * x236 + self.p.Jp_yy * \
            (beta1_dot * (-x233 - x234) + theta_dot * (x128 * x80 - x146 * x76) + x135 * x76) - self.p.Jp_zz * x236
        x238 = beta2_dot + x133
        x239 = x137 * x78
        x240 = x130 * x82
        x241 = x239 + x240
        x242 = x238 * x241
        x243 = self.p.Jp_xx * x242 + self.p.Jp_yy * \
            (beta2_dot * (-x239 - x240) + theta_dot * (x128 * x82 - x146 * x78) + x135 * x78) - self.p.Jp_zz * x242
        x244 = x130 * x76 - x137 * x80
        x245 = x232 * x244
        x246 = self.p.Jp_xx * (beta1_dot * x244 + theta_dot * (-x128 * x76 - \
                               x146 * x80) + x135 * x80) - self.p.Jp_yy * x245 + self.p.Jp_zz * x245
        x247 = x130 * x78 - x137 * x82
        x248 = x238 * x247
        x249 = self.p.Jp_xx * (beta2_dot * x247 + theta_dot * (-x128 * x78 - \
                               x146 * x82) + x135 * x82) - self.p.Jp_yy * x248 + self.p.Jp_zz * x248
        x250 = x138 * x2 - x164 * x46
        x251 = x161 * x250
        x252 = alphax_dot * x186
        x253 = phi_dot * x182
        x254 = -self.p.J3_xx * x251 + self.p.J3_yy * x251 + self.p.J3_zz * \
            (alphay_dot * x250 + theta_dot * (-x128 * x46 + x174 * x2) + x2 * x252 + x2 * x253)
        x255 = x138 * x46
        x256 = x164 * x2
        x257 = x255 + x256
        x258 = x161 * x257
        x259 = self.p.J3_xx * (alphay_dot * (-x255 - x256) + theta_dot * (-x128 * x2 - \
                               x174 * x46) - x252 * x46 - x253 * x46) - self.p.J3_yy * x258 + self.p.J3_zz * x258
        x260 = x130 * x137
        x261 = phi_dot * x154 - x124 * x127
        x262 = -self.p.J1_xx * x260 + self.p.J1_yy * x260 + self.p.J1_zz * x261
        x263 = x133 * x137
        x264 = self.p.J1_xx * x263 + self.p.J1_yy * (-x126 + x135) - self.p.J1_zz * x263
        x265 = 2 * self.p.lp * (-self.p.mp * x122 + self.p.mp * x132 -
                                x126 * x227 + x135 * x227 - x141 * x228)
        x266 = self.p.Jp_zz * x261
        x267 = x235 * x244
        x268 = -self.p.Jp_xx * x267 + self.p.Jp_yy * x267 + self.p.c_d * x212 + x266
        x269 = x241 * x247
        x270 = -self.p.Jp_xx * x269 + self.p.Jp_yy * x269 - self.p.c_d * x213 + x266
        x271 = x138 * x164
        x272 = alphax_dot * x164 + phi_dot * x180 + theta_dot * x175
        x273 = self.p.J2_xx * x271 + self.p.J2_yy * x272 - self.p.J2_zz * x271
        x274 = x138 * x160
        x275 = -self.p.J2_xx * x274 + self.p.J2_yy * x274 + \
            self.p.J2_zz * (theta_dot * x174 + x252 + x253)
        x276 = x123 + x140 + x147 + x149 + x152 + x155
        x277 = x250 * x257
        x278 = self.p.J3_xx * x277 + self.p.J3_yy * x272 - self.p.J3_zz * x277
        A[0, 0] = x1
        A[0, 1] = 0
        A[0, 2] = 0
        A[0, 3] = 0
        A[0, 4] = x7
        A[0, 5] = 0
        A[0, 6] = 0
        A[0, 7] = 0
        A[0, 8] = x29
        A[0, 9] = x42
        A[1, 0] = 0
        A[1, 1] = x1
        A[1, 2] = 0
        A[1, 3] = x43 + x45
        A[1, 4] = -x48
        A[1, 5] = 0
        A[1, 6] = 0
        A[1, 7] = x53
        A[1, 8] = x55
        A[1, 9] = x63
        A[2, 0] = 0
        A[2, 1] = 0
        A[2, 2] = x1
        A[2, 3] = x67
        A[2, 4] = x68
        A[2, 5] = 0
        A[2, 6] = 0
        A[2, 7] = x67
        A[2, 8] = x69
        A[2, 9] = x71
        A[3, 0] = 0
        A[3, 1] = 0
        A[3, 2] = 0
        A[3, 3] = 1
        A[3, 4] = 0
        A[3, 5] = 0
        A[3, 6] = 0
        A[3, 7] = 0
        A[3, 8] = 0
        A[3, 9] = 0
        A[4, 0] = 0
        A[4, 1] = 0
        A[4, 2] = 0
        A[4, 3] = 0
        A[4, 4] = 1
        A[4, 5] = 0
        A[4, 6] = 0
        A[4, 7] = 0
        A[4, 8] = 0
        A[4, 9] = 0
        A[5, 0] = 0
        A[5, 1] = 0
        A[5, 2] = 0
        A[5, 3] = 0
        A[5, 4] = 0
        A[5, 5] = 1
        A[5, 6] = 0
        A[5, 7] = 0
        A[5, 8] = 0
        A[5, 9] = 0
        A[6, 0] = 0
        A[6, 1] = 0
        A[6, 2] = 0
        A[6, 3] = 0
        A[6, 4] = 0
        A[6, 5] = 0
        A[6, 6] = 1
        A[6, 7] = 0
        A[6, 8] = 0
        A[6, 9] = 0
        A[7, 0] = 0
        A[7, 1] = x53
        A[7, 2] = x67
        A[7, 3] = x43 * x49 + x45 * x51 + x75
        A[7, 4] = -x48 * x51 + x65 * x68
        A[7, 5] = 0
        A[7, 6] = 0
        A[7, 7] = self.p.J1_xx + self.p.Jp_xx * x77 + self.p.Jp_xx * x79 + self.p.Jp_yy * \
            x81 + self.p.Jp_yy * x83 + self.p.m2 * x49**2 + self.p.m3 * x51**2 + x75 + x84
        A[7, 8] = x91
        A[7, 9] = x101
        A[8, 0] = x29
        A[8, 1] = x55
        A[8, 2] = x69
        A[8, 3] = -x44 * x54 + x90
        A[8, 4] = x102 + x103 * x24 * x4 + x103 * x26 * x6 + x27 * x7
        A[8, 5] = x104
        A[8, 6] = x104
        A[8, 7] = x91
        A[8, 8] = self.p.J1_yy * x105 + self.p.J1_zz * x106 + self.p.J2_yy * x110 + self.p.J2_zz * x111 + self.p.J3_xx * x111 * x73 + self.p.J3_yy * x110 + self.p.J3_zz * \
            x111 * x72 + self.p.m2 * x14**2 + self.p.m3 * x27**2 + x103 * x24**2 + x103 * x26**2 + x105 * x84 + x106 * x107 + x108 * x81 + x108 * x83 + x109 * x77 + x109 * x79
        A[8, 9] = x115
        A[9, 0] = x42
        A[9, 1] = x63
        A[9, 2] = x71
        A[9, 3] = x43 * x59 + x45 * x62 + x96
        A[9, 4] = self.p.J3_yy * x37 + x41 * x7 - x48 * x62 + x68 * x70
        A[9, 5] = x116
        A[9, 6] = x116
        A[9, 7] = x101
        A[9, 8] = x115
        A[9, 9] = self.p.J1_xx * x117 + self.p.J1_yy * x119 + self.p.J1_zz * x120 + self.p.J2_xx * x117 + self.p.J2_yy * x121 + self.p.J2_zz * x38**2 + self.p.J3_xx * x94**2 + self.p.J3_yy * x121 + self.p.J3_zz * x92**2 + self.p.Jp_xx * \
            x97**2 + self.p.Jp_xx * x98**2 + self.p.Jp_yy * x100**2 + self.p.Jp_yy * x99**2 + self.p.m2 * x35**2 + self.p.m2 * x59**2 + self.p.m3 * x41**2 + self.p.m3 * x62**2 + self.p.m3 * x70**2 + x107 * x120 + x117 * x74 + x117 * x84 + x119 * x84
        b[0] = self.p.m1 * x122 + v * x134 - w * x131 + x0 * x122 - x0 * x132 - x123 + \
            x126 * x9 + x133 * x142 - x135 * x9 - x140 - x147 - x149 - x152 - x155 - x189
        b[1] = -u * x134 + w * x195 - x0 * x193 + x0 * x196 - x133 * x203 - \
            x190 * x31 - x192 - x194 * x9 - x198 - x200 - x201 - x206 - x211
        b[2] = -2 * g * self.p.mp * x36 + u * x131 - v * x195 + x130 * x203 - x137 * \
            x142 - x190 * x36 + x212 + x213 - x214 - x215 - x216 - x217 - x218 - x219
        b[3] = x220 * (-alphax_dot + omega_x_cmd)
        b[4] = x220 * (-alphay_dot + omega_y_cmd)
        b[5] = x220 * (-beta1_dot + omega_1_cmd)
        b[6] = x220 * (-beta2_dot + omega_2_cmd)
        b[7] = -x2 * x259 - x211 * x51 - x219 * x65 - x222 - x223 - x225 - x226 + x229 - \
            x230 - x231 * x49 + x237 * x80 + x243 * x82 - x246 * x76 - x249 * x78 - x254 * x46
        b[8] = x11 * x262 + x11 * x268 + x11 * x270 - x113 * x254 + x114 * x259 - x14 * x276 - x189 * x27 - x20 * x273 - x20 * x278 + x211 * \
            x24 * x61 - x219 * x26 * x61 - x23 * x275 - x237 * x76 * x8 - x243 * x78 * x8 - x246 * x8 * x80 - x249 * x8 * x82 - x264 * x8 - x265 * x8
        b[9] = -x100 * x243 - x189 * x41 - x211 * x62 - x219 * x70 + x222 * x56 - x229 * x56 + x230 * x56 - x231 * x59 - x237 * x99 - x246 * x97 - x249 * x98 - x254 * \
            x92 - x259 * x94 - x262 * x36 - x264 * x31 - x265 * x31 - x268 * x36 - x270 * x36 - x273 * x37 - x275 * x38 - x276 * x35 - x278 * x37 + x56 * (x223 + x225 + x226)
        omega_dot = np.linalg.solve(A, b)

        return omega_dot

    def _compute_r_OSi(self, state):
        """computes center of mass locations of all bodies

        args:
            state (ModelState): current state

        Returns: list of x/y/z coordinates of center of mass of lower ball, upper ball, and lever arm.
        """
        [x, y, z] = state.pos
        [phi, theta, psi] = state.roll_pitch_yaw
        [alphax, alphay] = state.alpha

        I_r_OS1 = np.zeros(3)
        I_r_OS2 = np.zeros(3)
        I_r_OS3 = np.zeros(3)

        x0 = sin(phi)
        x1 = sin(psi)
        x2 = x0 * x1
        x3 = sin(theta)
        x4 = cos(phi)
        x5 = cos(psi)
        x6 = x4 * x5
        x7 = x2 + x3 * x6
        x8 = -self.p.l1 * x7 + x
        x9 = cos(alphax)
        x10 = x7 * x9
        x11 = sin(alphax)
        x12 = x1 * x4
        x13 = x0 * x5
        x14 = x11 * (-x12 + x13 * x3)
        x15 = x10 - x14
        x16 = x12 * x3 - x13
        x17 = -self.p.l1 * x16 + y
        x18 = x16 * x9
        x19 = x11 * (x2 * x3 + x6)
        x20 = x18 - x19
        x21 = cos(theta)
        x22 = x21 * x4
        x23 = -self.p.l1 * x22 + z
        x24 = x22 * x9
        x25 = x0 * x11 * x21
        x26 = x24 - x25
        x27 = sin(alphay)
        x28 = x21 * x27
        x29 = cos(alphay)
        I_r_OS1[0] = x
        I_r_OS1[1] = y
        I_r_OS1[2] = z
        I_r_OS2[0] = -self.p.l2 * x15 + x8
        I_r_OS2[1] = -self.p.l2 * x20 + x17
        I_r_OS2[2] = -self.p.l2 * x26 + x23
        I_r_OS3[0] = -self.p.l3 * x15 - self.p.l4 * (x10 * x29 - x14 * x29 + x28 * x5) + x8
        I_r_OS3[1] = -self.p.l3 * x20 - self.p.l4 * (x1 * x28 + x18 * x29 - x19 * x29) + x17
        I_r_OS3[2] = -self.p.l3 * x26 - self.p.l4 * (x24 * x29 - x25 * x29 - x27 * x3) + x23

        return [I_r_OS1, I_r_OS2, I_r_OS3]

    def _compute_body_visualization(self, center, l0, l1, q_IB):
        """computes visualization points of a body

        args:
            center (numpy.ndarray): center of the body [m]
            l0 : distance from center of mass to upper end of the body [m]
            l1 : length of the body [m]
            q_IB (Quaternion): orientation of the body frame B wrt. inertial frame I

        Returns: list of x/y/z coordinates of ball surface and zero angle reference
        """

        """https://stackoverflow.com/questions/44881885/python-draw-3d-cube"""
        points = np.array([[-1, -1, -1],
                           [1, -1, -1],
                           [1, 1, -1],
                           [-1, 1, -1],
                           [-1, -1, 1],
                           [1, -1, 1],
                           [1, 1, 1],
                           [-1, 1, 1]])

        width = 0.025

        P = np.dot(q_IB.rotation_matrix(), np.diag([width / 2, width / 2, l1 / 2]))

        Z = np.zeros((8, 3))
        for i in range(8):
            Z[i, :] = np.dot(P, points[i, :] + np.array([0, 0, l0 - 0.5 * l1])) + center

        # list of sides' polygons of figure
        verts = [
            [
                Z[0], Z[1], Z[2], Z[3]], [
                Z[4], Z[5], Z[6], Z[7]], [
                Z[0], Z[1], Z[5], Z[4]], [
                    Z[2], Z[3], Z[7], Z[6]], [
                        Z[1], Z[2], Z[6], Z[5]], [
                            Z[4], Z[7], Z[3], Z[0]], [
                                Z[2], Z[3], Z[7], Z[6]]]

        return verts

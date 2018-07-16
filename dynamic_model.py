"""Dynamic model of taketombo

author: Christof Dubs
"""
import numpy as np
from numpy import sin, cos, tan
from scipy.integrate import odeint
from rotation import Quaternion, quat_from_angle_vector


class ModelParam:
    """Physical parameters of taketombo

    Indices 1, 2 and 3 generally refer to the following rigid bodies:
        1: upper body (with propellers attached)
        2: middle body
        3: lower body


    Attributes:
        g: Gravitational constant [m/s^2]
        l1 : distance from center of mass of upper body to first motor's axis [m]
        l2 : distance from first motor's axis to center of mass of middle body [m]
        l3 : distance from center of mass of middle body to second motor's axis [m]
        l3 : distance from second motor's axis to center of mass of lower body [m]
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
        self.g = 9.81
        self.l1 = 0.025
        self.l2 = 0.025
        self.l3 = 0.050
        self.l4 = 0.025
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
        self.Jp_xx = 1e-8
        self.Jp_yy = 1e-8
        self.Jp_zz = 1e-5

    def is_valid(self,):
        """Checks validity of parameter configuration

        Returns:
            bool: True if valid, False if invalid.
        """
        return self.g > 0 and self.l1 > 0 and self.l2 > 0 and self.l3 > 0 and self.l4 > 0 and self.m1 > 0 and self.m2 > 0 and self.m3 > 0 and self.mp > 0 and self.tau > 0 and self.J1_xx > 0 and self.J1_yy > 0 and self.J1_zz > 0 and self.J2_xx > 0 and self.J2_yy > 0 and self.J2_zz > 0 and self.J3_xx > 0 and self.J3_yy > 0 and self.J3_zz > 0 and self.Jp_xx > 0 and self.Jp_yy > 0 and self.Jp_zz > 0


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

    @property
    def pos(self):
        return self.x[X_IDX:Z_IDX + 1]

    @pos.setter
    def pos(self, value):
        self.x[X_IDX:Z_IDX + 1] = value

    @property
    def vel(self):
        return self.x[U_IDX:PSI_DOT + 1]

    @vel.setter
    def vel(self, value):
        self.x[U_IDX:PSI_DOT + 1] = value

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

    def simulate_step(self, delta_t, omega_cmd):
        """Simulate one time step

        Simulates the changes of the state over a time interval.

        args:
            delta_t: time step [s]
            omega_cmd (np.ndarray): motor speed commands [rad/s]
        """
        t = np.array([0, delta_t])
        self.state.x = odeint(self._x_dot, self.state.x, t, args=(omega_cmd,))[-1]

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
            r_OSi[0], self.p.l1, 2 * self.p.l1, Quaternion(state.q1))
        vis['middle_body'] = self._compute_body_visualization(
            r_OSi[1], self.p.l2, self.p.l3, state.q2)
        vis['lower_body'] = self._compute_body_visualization(
            r_OSi[2], self.p.l4, 2 * self.p.l4, state.q3)
        return vis

    def _x_dot(self, x, t, omega_cmd):
        """computes the derivative of the state

        This function returns an numpy.array of the derivatives of the states, given the current state and inputs.

        Its signature is compatible with scipy.integrate.odeint's first callable argument.

        args:
            x (numpy.ndarray): state at which the state derivative function is evaluated
            t: time [s]. Since this system is time invariant, this argument is unused.
            omega_cmd (np.ndarray): motor speed commands [rad/s]
        returns:
            ModelState containing the time derivatives of all states
        """
        eval_state = ModelState(x, skip_checks=True)

        xdot = ModelState()

        xdot.vel = self._compute_vel_dot(eval_state, omega_cmd)

        omega_1 = self._get_upper_body_omega(eval_state)
        xdot.q1 = Quaternion(eval_state.q1).q_dot(omega_1, frame='body')

        R_IB = eval_state.q1.rotation_matrix()

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

    def _compute_vel_dot(self, state, omega_cmd):
        """computes derivative of velocity states

        args:
            state (ModelState): current state
            omega_cmd (np.ndarray): motor speed commands [rad/s]

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

        [omega_x_cmd, omega_y_cmd] = omega_cmd

        A = np.zeros([10, 10])
        b = np.zeros(10)

        # auto-generated symbolic expressions
        x0 = self.p.m1 + self.p.m2 + self.p.m3
        x1 = cos(alphay)
        x2 = self.p.l4 * x1
        x3 = sin(alphax)
        x4 = x3**2
        x5 = cos(alphax)
        x6 = self.p.m3 * (-x2 * x4 - x2 * x5**2)
        x7 = sin(phi)
        x8 = cos(theta)
        x9 = self.p.l1 * x8
        x10 = -x7 * x9
        x11 = cos(phi)
        x12 = x11 * x3
        x13 = x12 * x8
        x14 = self.p.l2 * x5
        x15 = x7 * x8
        x16 = -self.p.l2 * x13 + x10 - x14 * x15
        x17 = self.p.l3 * x11 * x3
        x18 = self.p.l3 * x5
        x19 = x2 * x3
        x20 = x5 * x7
        x21 = x13 + x20 * x8
        x22 = x11 * x5
        x23 = x3 * x7
        x24 = x22 * x8 - x23 * x8
        x25 = x21 * x3 + x24 * x5
        x26 = x2 * x5
        x27 = x21 * x5 - x24 * x3
        x28 = x10 - x15 * x18 - x17 * x8 - x19 * x25 - x26 * x27
        x29 = self.p.m2 * x16 + self.p.m3 * x28
        x30 = -self.p.l1 * x11
        x31 = self.p.l2 * x23 - x11 * x14 + x30
        x32 = self.p.m2 * x31
        x33 = self.p.l3 * x3
        x34 = x22 - x23
        x35 = -x12 - x20
        x36 = x3 * x34 + x35 * x5
        x37 = -x3 * x35 + x34 * x5
        x38 = -x11 * x18 - x19 * x36 - x26 * x37 + x30 + x33 * x7
        x39 = self.p.m3 * x38
        x40 = x32 + x39
        x41 = self.p.m2 * x14
        x42 = x18 + x26
        x43 = self.p.m3 * x42
        x44 = sin(alphay)
        x45 = self.p.l4 * self.p.m3 * x44
        x46 = x3 * x45
        x47 = self.p.l1 + x14
        x48 = self.p.m2 * x47
        x49 = self.p.l1 + x42
        x50 = self.p.m3 * x49
        x51 = x48 + x50
        x52 = sin(theta)
        x53 = self.p.l1 * x52
        x54 = -x53
        x55 = -x14 * x52 + x54
        x56 = self.p.l4 * x1 * x52
        x57 = self.p.l4 * x44
        x58 = -x18 * x52 - x25 * x57 - x5 * x56 + x54
        x59 = self.p.m2 * x55 + self.p.m3 * x58
        x60 = x36 * x45
        x61 = -x60
        x62 = self.p.l2 * self.p.m2 * x3
        x63 = x19 + x33
        x64 = self.p.m3 * x63
        x65 = x62 + x64
        x66 = x45 * x5
        x67 = x27 * x57 - x3 * x56 - x33 * x52
        x68 = self.p.m3 * x67 - x52 * x62
        x69 = x37 * x45
        x70 = x1**2
        x71 = x44**2
        x72 = self.p.l2**2 * self.p.m2 * x4
        x73 = self.p.J2_xx + self.p.J3_xx * x70 + self.p.J3_zz * x71 + self.p.m3 * x63**2 + x72
        x74 = x1 * x24 - x44 * x52
        x75 = self.p.J3_zz * x74
        x76 = -x1 * x52 - x24 * x44
        x77 = self.p.J3_xx * x76
        x78 = -self.p.J2_xx * x52 + x1 * x77 + x44 * x75 - x52 * x72 + x64 * x67
        x79 = -self.p.J1_xx * x52 + x48 * x55 + x50 * x58 + x78
        x80 = x1 * x35 * x44
        x81 = -self.p.J3_xx * x80 + self.p.J3_zz * x80 + x63 * x69
        x82 = -x49 * x60 + x81
        x83 = x52**2
        x84 = x7**2
        x85 = x8**2
        x86 = x11**2
        x87 = x21**2
        x88 = x11 * x7 * x8
        x89 = self.p.J3_yy * x34
        x90 = x1 * x35
        x91 = x35 * x44
        x92 = self.p.J1_yy * x88 - self.p.J1_zz * x88 + self.p.J2_yy * x21 * x34 + self.p.J2_zz * \
            x24 * x35 + x16 * x32 + x21 * x89 + x28 * x39 - x58 * x60 + x67 * x69 + x75 * x90 - x77 * x91
        x93 = self.p.l4**2 * self.p.m3 * x71
        x94 = x34**2
        x95 = x35**2
        x96 = g * x52
        x97 = -self.p.m2 * x96
        x98 = theta_dot * x11
        x99 = psi_dot * x8
        x100 = x7 * x99
        x101 = x100 + x98
        x102 = self.p.m1 * x101
        x103 = theta_dot * x7
        x104 = -x103 + x11 * x99
        x105 = self.p.m1 * x104
        x106 = self.p.l2 * x3
        x107 = psi_dot * x52
        x108 = phi_dot - x107
        x109 = alphax_dot + x108
        x110 = self.p.m2 * (w + x106 * x109)
        x111 = x101 * x110
        x112 = self.p.m2 * theta_dot
        x113 = psi_dot * x53 * x7
        x114 = x107 * x12
        x115 = psi_dot * x52 * x7
        x116 = x112 * (self.p.l2 * x114 + x113 + x115 * x14)
        x117 = -x104 * x14
        x118 = alphax_dot * self.p.m2 * (x101 * x106 + x117)
        x119 = self.p.l1 * x108
        x120 = self.p.m2 * (v + x109 * x14 + x119)
        x121 = -x104 * x120
        x122 = -self.p.l1 * x104
        x123 = -x100 - x98
        x124 = phi_dot * self.p.m2 * (-x106 * x123 + x117 + x122)
        x125 = x109 * x33
        x126 = x109 * x19
        x127 = x101 * x5
        x128 = x104 * x3
        x129 = x127 + x128
        x130 = alphay_dot + x129
        x131 = x130 * x5
        x132 = x104 * x5
        x133 = -x101 * x3 + x132
        x134 = x131 - x133 * x3
        x135 = x134 * x57
        x136 = self.p.m3 * (w + x125 + x126 + x135)
        x137 = x109 * x18 + x109 * x26
        x138 = x130 * x3
        x139 = x133 * x5 + x138
        x140 = x139 * x57
        x141 = self.p.m3 * (v + x119 + x137 - x140)
        x142 = self.p.m3 * theta_dot
        x143 = -x107 * x22 + x107 * x23
        x144 = -x107 * x20 - x114
        x145 = x143 * x5 + x144 * x3
        x146 = -x143 * x3 + x144 * x5
        x147 = alphay_dot * self.p.m3
        x148 = -x104 * x18
        x149 = x123 * x3 + x132
        x150 = -x128
        x151 = x123 * x5 + x150
        x152 = x149 * x3 + x151 * x5
        x153 = x149 * x5 - x151 * x3
        x154 = alphax_dot * self.p.m3
        x155 = -x127 + x150
        x156 = x131 + x155 * x5
        x157 = -x138 - x155 * x3
        x158 = phi_dot * self.p.m3 * (x122 - x123 * x33 + x148 - x152 * x19 - x153 * x26) - self.p.m3 * x96 + x101 * x136 - x104 * x141 + x142 * (
            x107 * x17 + x113 + x115 * x18 - x145 * x19 - x146 * x26) + x147 * (x135 * x5 + x140 * x3) + x154 * (x101 * x33 + x134 * x19 - x139 * x26 + x148 - x156 * x19 - x157 * x26)
        x159 = g * self.p.m1 * x8
        x160 = g * self.p.m2 * x8
        x161 = x160 * x7
        x162 = self.p.m1 * x108
        x163 = alphax_dot * x109
        x164 = -x163 * x62
        x165 = -psi_dot * x9
        x166 = x112 * (-x14 * x99 + x165)
        x167 = -x108 * x110
        x168 = -self.p.l1 * x101 + u
        x169 = self.p.m2 * (-x101 * x14 - x104 * x106 + x168)
        x170 = x104 * x169
        x171 = g * self.p.m3 * x8
        x172 = phi_dot * self.p.l4 * self.p.m3 * x44
        x173 = self.p.l4 * x109 * x44
        x174 = self.p.m3 * (-x101 * x18 - x104 * x33 - x134 * x26 - x139 * x19 + x168)
        x175 = x104 * x174 - x108 * x136 + x142 * (-x145 * x57 + x165 - x18 * x99 - x26 * x99) + x147 * (
            -x139 * x2 - x173 * x5) - x152 * x172 + x154 * (-x125 - x126 - x156 * x57) + x171 * x7
        x176 = x11 * x160
        x177 = psi_dot * theta_dot * x8
        x178 = -x177 * x62
        x179 = x163 * x41
        x180 = x108 * x120
        x181 = -x101 * x169
        x182 = -x101 * x174 + x108 * x141 + x11 * x171 + x142 * \
            (x146 * x57 - x19 * x99 - x33 * x99) + x147 * (x134 * x2 - x173 * x3) + x153 * x172 + x154 * (x137 + x157 * x57)
        x183 = 1 / self.p.tau
        x184 = x129 * x133
        x185 = -self.p.J2_xx * x177 - self.p.J2_yy * x184 + self.p.J2_zz * x184
        x186 = -self.p.J1_xx * x177
        x187 = x101 * x104
        x188 = self.p.J1_zz * x187
        x189 = -self.p.J1_yy * x187
        x190 = x106 * (x176 + x178 + x179 + x180 + x181)
        x191 = x161 + x164 + x166 + x167 + x170
        x192 = x1 * x109 - x133 * x44
        x193 = x130 * x192
        x194 = alphax_dot * x155
        x195 = phi_dot * x151
        x196 = -self.p.J3_xx * x193 + self.p.J3_yy * x193 + self.p.J3_zz * \
            (alphay_dot * x192 + theta_dot * (x1 * x143 - x44 * x99) + x1 * x194 + x1 * x195)
        x197 = x109 * x44
        x198 = x1 * x133
        x199 = x197 + x198
        x200 = x130 * x199
        x201 = self.p.J3_xx * (alphay_dot * (-x197 - x198) + theta_dot * (-x1 * x99 - \
                               x143 * x44) - x194 * x44 - x195 * x44) - self.p.J3_yy * x200 + self.p.J3_zz * x200
        x202 = x101 * x108
        x203 = -self.p.J1_xx * x202 + self.p.J1_yy * x202 + \
            self.p.J1_zz * (phi_dot * x123 - x107 * x98)
        x204 = x104 * x108
        x205 = self.p.J1_xx * x204 + self.p.J1_yy * \
            (phi_dot * x104 - x103 * x107) - self.p.J1_zz * x204
        x206 = x109 * x129
        x207 = -self.p.J2_xx * x206 + self.p.J2_yy * x206 + \
            self.p.J2_zz * (theta_dot * x143 + x194 + x195)
        x208 = x109 * x133
        x209 = alphax_dot * x133 + phi_dot * x149 + theta_dot * x144
        x210 = self.p.J2_xx * x208 + self.p.J2_yy * x209 - self.p.J2_zz * x208
        x211 = x111 + x116 + x118 + x121 + x124 + x97
        x212 = x192 * x199
        x213 = self.p.J3_xx * x212 + self.p.J3_yy * x209 - self.p.J3_zz * x212
        A[0, 0] = x0
        A[0, 1] = 0
        A[0, 2] = 0
        A[0, 3] = 0
        A[0, 4] = x6
        A[0, 5] = 0
        A[0, 6] = 0
        A[0, 7] = 0
        A[0, 8] = x29
        A[0, 9] = x40
        A[1, 0] = 0
        A[1, 1] = x0
        A[1, 2] = 0
        A[1, 3] = x41 + x43
        A[1, 4] = -x46
        A[1, 5] = 0
        A[1, 6] = 0
        A[1, 7] = x51
        A[1, 8] = x59
        A[1, 9] = x61
        A[2, 0] = 0
        A[2, 1] = 0
        A[2, 2] = x0
        A[2, 3] = x65
        A[2, 4] = x66
        A[2, 5] = 0
        A[2, 6] = 0
        A[2, 7] = x65
        A[2, 8] = x68
        A[2, 9] = x69
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
        A[5, 5] = 0
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
        A[6, 6] = 0
        A[6, 7] = 0
        A[6, 8] = 0
        A[6, 9] = 0
        A[7, 0] = 0
        A[7, 1] = x51
        A[7, 2] = x65
        A[7, 3] = x41 * x47 + x43 * x49 + x73
        A[7, 4] = -x46 * x49 + x63 * x66
        A[7, 5] = 0
        A[7, 6] = 0
        A[7, 7] = self.p.J1_xx + self.p.m2 * x47**2 + self.p.m3 * x49**2 + x73
        A[7, 8] = x79
        A[7, 9] = x82
        A[8, 0] = x29
        A[8, 1] = x59
        A[8, 2] = x68
        A[8, 3] = x41 * x55 + x43 * x58 + x78
        A[8, 4] = self.p.J3_yy * x21 + x28 * x6 - x46 * x58 + x66 * x67
        A[8, 5] = 0
        A[8, 6] = 0
        A[8, 7] = x79
        A[8, 8] = self.p.J1_xx * x83 + self.p.J1_yy * x84 * x85 + self.p.J1_zz * x85 * x86 + self.p.J2_xx * x83 + self.p.J2_yy * x87 + self.p.J2_zz * x24**2 + self.p.J3_xx * \
            x76**2 + self.p.J3_yy * x87 + self.p.J3_zz * x74**2 + self.p.m2 * x16**2 + self.p.m2 * x55**2 + self.p.m3 * x28**2 + self.p.m3 * x58**2 + self.p.m3 * x67**2 + x72 * x83
        A[8, 9] = x92
        A[9, 0] = x40
        A[9, 1] = x61
        A[9, 2] = x69
        A[9, 3] = -x42 * x60 + x81
        A[9, 4] = x3 * x36 * x93 + x37 * x5 * x93 + x38 * x6 + x89
        A[9, 5] = 0
        A[9, 6] = 0
        A[9, 7] = x82
        A[9, 8] = x92
        A[9, 9] = self.p.J1_yy * x86 + self.p.J1_zz * x84 + self.p.J2_yy * x94 + self.p.J2_zz * x95 + self.p.J3_xx * x71 * x95 + \
            self.p.J3_yy * x94 + self.p.J3_zz * x70 * x95 + self.p.m2 * x31**2 + self.p.m3 * x38**2 + x36**2 * x93 + x37**2 * x93
        b[0] = self.p.m1 * x96 + v * x105 - w * x102 - x111 - x116 - x118 - x121 - x124 - x158 - x97
        b[1] = -u * x105 + w * x162 - x159 * x7 - x161 - x164 - x166 - x167 - x170 - x175
        b[2] = u * x102 - v * x162 - x11 * x159 - x176 - x178 - x179 - x180 - x181 - x182
        b[3] = x183 * (-alphax_dot + omega_x_cmd)
        b[4] = x183 * (-alphay_dot + omega_y_cmd)
        b[5] = 0
        b[6] = 0
        b[7] = -x1 * x201 - x175 * x49 - x182 * x63 - x185 - \
            x186 - x188 - x189 - x190 - x191 * x47 - x196 * x44
        b[8] = -x11 * x203 * x8 - x15 * x205 - x158 * x28 - x16 * x211 - x175 * x58 - x182 * x67 + x185 * x52 + x190 * \
            x52 - x191 * x55 - x196 * x74 - x201 * x76 - x207 * x24 - x21 * x210 - x21 * x213 + x52 * (x186 + x188 + x189)
        b[9] = -x11 * x205 - x158 * x38 + x175 * x36 * x57 - x182 * x37 * x57 - x196 * \
            x90 + x201 * x91 + x203 * x7 - x207 * x35 - x210 * x34 - x211 * x31 - x213 * x34

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
        p1 = np.array([0, 0, -l0])
        p2 = np.array([0, 0, l1 - l0])

        R_IB = q_IB.rotation_matrix()

        p_rot = np.dot(R_IB, np.array([p1, p2]).T)

        return [np.array([center[i] + p_rot[i, :]]) for i in range(3)]

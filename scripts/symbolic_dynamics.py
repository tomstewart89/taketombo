"""Script to generate symbolic dynamics of taketombo

author: Christof Dubs
"""
from sympy import *


# position
x, y, z = symbols('x, y, z')

# velocity
u, v, w = symbols('u, v, w')
x_dot, y_dot, z_dot = symbols('x_dot, y_dot, z_dot')

# accelerations
u_dot, v_dot, w_dot = symbols('u_dot, v_dot, w_dot')

# angles
alphax, alphay, beta1, beta2, phi, psi, theta = symbols(
    'alphax, alphay, beta1, beta2, phi, psi, theta')

# angular velocities
alphax_dot, alphay_dot, beta1_dot, beta2_dot, phi_dot, psi_dot, theta_dot = symbols(
    'alphax_dot, alphay_dot, beta1_dot, beta2_dot, phi_dot, psi_dot, theta_dot')

# angular accelerations
alphax_ddot, alphay_ddot, beta1_ddot, beta2_ddot, phi_ddot, psi_ddot, theta_ddot = symbols(
    'alphax_ddot, alphay_ddot, beta1_ddot, beta2_ddot, phi_ddot, psi_ddot, theta_ddot')

# parameter
l1, l2, l3, l4, m1, m2, m3, mp, tau, J1_xx, J1_yy, J1_zz, J2_xx, J2_yy, J2_zz, J3_xx, J3_yy, J3_zz, Jp_xx, Jp_yy, Jp_zz = symbols(
    'l1, l2, l3, l4, m1, m2, m3, mp, tau, J1_xx, J1_yy, J1_zz, J2_xx, J2_yy, J2_zz, J3_xx, J3_yy, J3_zz, Jp_xx, Jp_yy, Jp_zz')

# constants
g = symbols('g')

# inputs
Tx, Ty = symbols('Tx Ty')
omega_x_cmd, omega_y_cmd = symbols('omega_x_cmd omega_y_cmd')
omega_cmd = Matrix([omega_x_cmd, omega_y_cmd])

# parameter lists:
m = [m1, m2, m3, mp, mp]
J = [diag(J1_xx, J1_yy, J1_zz),  diag(J2_xx, J2_yy, J2_zz),  diag(
    J3_xx, J3_yy, J3_zz), diag(Jp_xx, Jp_yy, Jp_zz), diag(Jp_xx, Jp_yy, Jp_zz)]

# primary (upper) body
R_IB1 = rot_axis3(-psi) * rot_axis2(-theta) * rot_axis1(-phi)
I_r_OS1 = Matrix([x, y, z])
B1_v_S1 = Matrix([u, v, w])
I_v_S1 = R_IB1*B1_v_S1
[x_dot, y_dot, z_dot] = I_v_S1

B1_omega_IB1 = Matrix([phi_dot, 0, 0]) + rot_axis1(phi) * Matrix(
    [0, theta_dot, 0]) + rot_axis1(phi) * rot_axis2(theta) * Matrix([0, 0, psi_dot])

# middle body:
R_B1B2 = rot_axis1(-alphax)
R_IB2 = R_IB1*R_B1B2
B1_r_S1M1 = Matrix([0, 0, -l1])

B1_v_M1 = B1_v_S1 + B1_omega_IB1.cross(B1_r_S1M1)

B1_omega_IB2 = B1_omega_IB1 + Matrix([alphax_dot, 0, 0])

B2_r_M1S2 = Matrix([0, 0, -l2])

B1_v_S2 = B1_v_M1 + B1_omega_IB2.cross(R_B1B2*B2_r_M1S2)

I_r_OS2 = I_r_OS1 + R_IB1*B1_r_S1M1 + R_IB2*B2_r_M1S2
B2_omega_IB2 = R_B1B2.T * B1_omega_IB2


# lower body
R_B2B3 = rot_axis2(-alphay)
R_B1B3 = R_B1B2*R_B2B3
R_IB3 = R_IB1*R_B1B3
B2_r_M1M2 = Matrix([0, 0, -l3])

B1_v_M2 = B1_v_M1 + B1_omega_IB2.cross(R_B1B2*B2_r_M1M2)

B2_omega_IB3 = B2_omega_IB2 + Matrix([0, alphay_dot, 0])
B1_omega_IB3 = R_B1B2*B2_omega_IB3
B3_r_M2S3 = Matrix([0, 0, -l4])

B1_v_S3 = B1_v_M2 + B1_omega_IB3.cross(R_B1B3*B3_r_M2S3)

I_r_OS3 = I_r_OS1 + R_IB1*B1_r_S1M1 + R_IB2*B2_r_M1M2 + R_IB3*B3_r_M2S3
B3_omega_IB3 = R_B2B3.T * B2_omega_IB3

# todo: propeller1
# todo: propeller2

# calculate Jacobians
B1_v_i = [B1_v_S1, B1_v_S2, B1_v_S3]
Bi_om_i = [B1_omega_IB1, B2_omega_IB2, B3_omega_IB3]

acc = Matrix([u_dot, v_dot, w_dot, alphax_ddot, alphay_ddot,
              beta1_ddot, beta2_ddot, phi_ddot, psi_ddot, theta_ddot])
vel = Matrix([u, v, w, alphax_dot, alphay_dot, beta1_dot,
              beta2_dot, phi_dot, psi_dot, theta_dot])
pos = Matrix([x, y, z, alphax, alphay, beta1, beta2, phi, psi, theta])
pos_dot = Matrix([x_dot, y_dot, z_dot, alphax_dot, alphay_dot,
                  beta1_dot, beta2_dot, phi_dot, psi_dot, theta_dot])

B1_J_i = [v.jacobian(vel) for v in B1_v_i]
Bi_JR_i = [om.jacobian(vel) for om in Bi_om_i]

# Forces
B1_F1 = R_IB1.T*Matrix([0, 0, -m1 * g])
B1_F2 = R_IB1.T*Matrix([0, 0, -m2 * g])
B1_F3 = R_IB1.T*Matrix([0, 0, -m3 * g])
B1_F_i = [B1_F1, B1_F2, B1_F3]

Bi_M1 = Matrix([-Tx, 0, 0])
Bi_M2 = Matrix([Tx, -Ty, 0])
Bi_M3 = Matrix([0, Ty, 0])

Bi_M_i = [Bi_M1, Bi_M2, Bi_M3]

# Impulse
B1_p_i = [m[i] * B1_v_i[i] for i in range(3)]
B1_p_dot_i = [p.jacobian(vel) * acc + p.jacobian(pos)
              * pos_dot + B1_omega_IB1.cross(p) for p in B1_p_i]

# Spin
omega_diff_i = [om.jacobian(vel) * acc + om.jacobian(pos)
                * pos_dot for om in Bi_om_i]
Bi_NS_dot_i = [J[i] * omega_diff_i[i] +
               Bi_om_i[i].cross(J[i] * Bi_om_i[i]) for i in range(3)]

# dynamics
print('generating dynamics')
dyn = zeros(10, 1)
for i in range(3):
    dyn += B1_J_i[i].T * (B1_p_dot_i[i] - B1_F_i[i]) + \
        Bi_JR_i[i].T * (Bi_NS_dot_i[i] - Bi_M_i[i])
    print('generated term {} of 3 dynamic terms'.format(i))

# replace the 4th and 5th equation (the only ones containing T)
dyn[3] = alphax_ddot - 1 / tau * (omega_x_cmd - alphax_dot)
dyn[4] = alphay_ddot - 1 / tau * (omega_y_cmd - alphay_dot)

# check that all Tx, Ty terms are eliminated
print(Matrix(dyn[:]).jacobian(Matrix([Tx, Ty])) == zeros(10, 2))

# set Tx, Ty to zero directly instead of simplifying (terms can be ... + Tx + ... - Tx)
dyn = dyn.subs([('Tx', 0), ('Ty', 0)])

A = dyn.jacobian(acc)
b = dyn.subs([(x, 0) for x in acc])

common_sub_expr = cse([A, b])

sub_list = [
    (x,
     'self.p.' +
     x) for x in [
        'l1', 'l2', 'l3', 'l4', 'm1', 'm2', 'm3', 'mp', 'tau', 'J1_xx', 'J1_yy', 'J1_zz', 'J2_xx', 'J2_yy', 'J2_zz', 'J3_xx', 'J3_yy', 'J3_zz', 'Jp_xx', 'Jp_yy', 'Jp_zz']]

for term in common_sub_expr[0]:
    print('        {} = {}'.format(term[0], term[1].subs(sub_list)))

for row in range(A.rows):
    for col in range(A.cols):
        print('        A[{},{}] = {}'.format(row, col,
                                             common_sub_expr[1][0][row, col].subs(sub_list)))

for row in range(b.rows):
    print('        b[{}] = {}'.format(
        row, -common_sub_expr[1][1][row].subs(sub_list)))

# kinematic relations
common_sub_expr = cse(B1_omega_IB1)
for term in common_sub_expr[0]:
    print('        {} = {}'.format(term[0], term[1].subs(sub_list)))

for row in range(B1_omega_IB1.rows):
    print('        B1_omega_IB1[{}] = {}'.format(
        row, common_sub_expr[1][0][row].subs(sub_list)))

# position vectors
common_sub_expr = cse([I_r_OS1, I_r_OS2, I_r_OS3])
for term in common_sub_expr[0]:
    print('        {} = {}'.format(term[0], term[1].subs(sub_list)))

for row in range(I_r_OS1.rows):
    print('        I_r_OS1[{}] = {}'.format(
        row, common_sub_expr[1][0][row].subs(sub_list)))

for row in range(I_r_OS2.rows):
    print('        I_r_OS2[{}] = {}'.format(
        row, common_sub_expr[1][1][row].subs(sub_list)))

for row in range(I_r_OS3.rows):
    print('        I_r_OS3[{}] = {}'.format(
        row, common_sub_expr[1][2][row].subs(sub_list)))

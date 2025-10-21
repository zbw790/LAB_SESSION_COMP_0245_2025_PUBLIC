import numpy as np


def feedback_lin_ctrl(dyn_model, q_, qd_, q_d, qd_d, kp, kd):
    """
    Perform feedback linearization control on a robotic system.

    Parameters:
    - dyn_model (pin_wrapper): The dynamics model of the robot encapsulated within a 'pin_wrapper' object,
                               which provides methods for computing robot dynamics such as mass matrices,
                               Coriolis forces, etc.
    - q_ (numpy.ndarray): Measured positions of the robot's joints, indicating the current actual positions
                          as measured by sensors or estimated by observers.
    - qd_ (numpy.ndarray): Measured velocities of the robot's joints, reflecting the current actual velocities.
    - q_d (numpy.ndarray): Desired positions for the robot's joints set by a trajectory generator or a higher-level
                           controller, dictating target positions.
    - qd_d (numpy.ndarray): Desired velocities for the robot's joints, specifying the rate at which the joints
                            should move towards their target positions.
    - kp (float or numpy.ndarray): Proportional gain(s) for the control system, which can be a uniform value across
                                   all joints or unique for each joint, adjusting the response to position error.
    - kd (float or numpy.ndarray): Derivative gain(s), similar to kp, affecting the response to velocity error and
                                   aiding in system stabilization by damping oscillations.

    Returns:
    None

    This function computes the control inputs necessary to achieve desired joint positions and velocities by
    applying feedback linearization, using the robot's dynamic model to appropriately compensate for its
    inherent dynamics. The control law implemented typically combines proportional-derivative (PD) control
    with dynamic compensation to achieve precise and stable motion.
    """
    # here i want to add a control that if the kp gains are a numpy vector ill check if the size is good
    #  and then i will create the P matrix and D matrix using the vector a diagonal of the matrices
    # if the size is not good i will raise an error
    # if the kp is a scalar i will create a diagonal matrix with the scalar value
    # same for kd
    if isinstance(kp, np.ndarray):
        if kp.size != dyn_model.getNumberofActuatedJoints():
            raise ValueError("The size of the kp vector is not correct") 
    else:
        kp = np.array([kp] * dyn_model.getNumberofActuatedJoints())
    
    P = np.diag(kp)
    
    if isinstance(kd, np.ndarray):
        if kd.size != dyn_model.getNumberofActuatedJoints():
            raise ValueError("The size of the kd vector is not correct")
    else:
        kd = np.array([kd] * dyn_model.getNumberofActuatedJoints())
    D = np.diag(kd)
 
    # here i compute the feeback linearization tau // the reordering is already done inside compute all teamrs
    dyn_model.ComputeAllTerms(q_, qd_)
    # reoder measurements in pinocchio order
    q_mes = dyn_model.ReoderJoints2PinVec(q_,"pos")
    qd_mes = dyn_model.ReoderJoints2PinVec(qd_,"vel")
    # control 
    u = P @ (q_d - q_mes) + D @ (qd_d - qd_mes)
    n = dyn_model.res.c + dyn_model.res.g
    
    tau_FL = dyn_model.res.M @ u + n

    tau_FL = dyn_model._FromPinToExtVec(tau_FL)
  
    return tau_FL
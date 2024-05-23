#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <Eigen/Sparse>
#include <chrono>


using namespace Eigen;

typedef Matrix<double, 3, 3> Matrix3d;
typedef Matrix<double, 3, 1> Vector3d;
typedef std::vector<Matrix3d, aligned_allocator<Matrix3d>> PosesVec;

struct Jacobian {
    Matrix3d i;
    Matrix3d j;
    Vector3d e;
    Jacobian(const Matrix3d _i, const Matrix3d _j, const Vector3d _e) : i(_i), j(_j), e(_e) {}
};

Matrix<double, 3, Dynamic> Traj(3, 12);

Matrix3d v2t(Vector3d v){
    Matrix3d T = Matrix3d::Zero();
    T << std::cos(v(2)), -std::sin(v(2)), v(0),
         std::sin(v(2)), std::cos(v(2)), v(1),
                      0,              0,    1;
    return T;
}

Vector3d t2v(Matrix3d T){
    Vector3d v = Vector3d::Zero();
    v(0) = T(0, 2);
    v(1) = T(1, 2);
    v(2) = std::atan2(T(1, 0), T(0, 0));
    return v;
}


/*
Use dead reckoning to find initial poses for graph optimization
Output:
    x: a vector of shape(36, 1) containing 12 initial poses
*/
Matrix<double, 36, 1> get_initial_poses(){
    PosesVec Poses;
    Matrix3d initial_pose = Matrix3d::Identity(); // initial pose
    Poses.push_back(initial_pose);

    // dead reckoning
    // drop the last frame, only 12 frames
    for (size_t i = 0; i < 11; ++i) {
        Vector3d pose = Traj.col(i);
        Matrix3d transformation;
        transformation << std::cos(pose(2)), -std::sin(pose(2)), pose(0),
                          std::sin(pose(2)), std::cos(pose(2)),  pose(1),
                          0,                  0,                   1;
        Matrix3d new_pose = Poses.back() * transformation;
        Poses.push_back(new_pose);
    }

    Matrix<double, 36, 1> x;
    // transform all poses to vector form
    for (size_t i = 0; i < Poses.size(); ++i) {
        Eigen::Vector3d v = t2v(Poses[i]);
        x(i * 3) = v(0);
        x(i * 3 + 1) = v(1);
        x(i * 3 + 2) = v(2);
    }

    return x;
}

/*
Inputs:
    x: 36x1 vector, each 3 elements represent the current guess pose of that frame.
    z: the relative transform vector from frame i to frame j
    i: the frame id of i
    j: the frame id of j
Outputs:
    I: Jacobian matrix of vector e with respect to vector xi
    J: Jacobian matrix of vector e with respect to vector xj
    e: vector e
*/
Jacobian cal_jMatrix(Matrix<double, 36, 1> x, Vector3d z, int i, int j){
    Matrix3d I = Matrix3d::Zero();
    Matrix3d J = Matrix3d::Zero();
    Vector3d e = Vector3d::Zero();

    Map<Eigen::Matrix<double, 3, 12>> x_m(x.data());

    Matrix3d Ti = v2t(x_m.col(i));
    Matrix2d Ri = Ti.block<2,2>(0,0);
    Vector2d ti = Ti.block<2,1>(0,2);

    Matrix3d Tj = v2t(x_m.col(j));
    Matrix2d Rj = Tj.block<2,2>(0,0);
    Vector2d tj = Tj.block<2,1>(0,2);

    Matrix3d T_ij = v2t(z);
    Matrix2d R_ij = T_ij.block<2,2>(0,0);
    Vector2d t_ij = T_ij.block<2,1>(0,2);
    
    double theta_z = z(2);
    double theta_i = t2v(Ti)(2);
    double theta_j = t2v(Tj)(2);

    Matrix2d partial_RiT_theta;
    partial_RiT_theta << -std::sin(theta_i), std::cos(theta_i),
                        -std::cos(theta_i), -std::sin(theta_i);

    I.block<2, 2>(0, 0) = - R_ij.transpose() * Ri.transpose();
    I.block<2, 1>(0, 2) = R_ij.transpose() * partial_RiT_theta * (tj - ti);
    I(2, 2) = -1;

    J.block<2, 2>(0, 0) = R_ij.transpose() * Ri.transpose();
    J(2, 2) = 1;

    e.block<2,1>(0, 0) = R_ij.transpose() * ((Ri.transpose() * (tj - ti)) - t_ij);
    e(2) = theta_j - theta_i - theta_z;

    if (e(2) > M_PI)
        e(2) -= 2 * M_PI;
    else if (e(2) < -M_PI)
        e(2) += 2 * M_PI;

    Jacobian J_ij(I, J, e);
    return J_ij;
}


int main() {
    Traj <<
        -0.146303, -0.140469, -0.153574, -0.124153, -0.119121, -0.104924, -0.138374, -0.127593, -0.129433, -0.157566, -0.116292, -0.137706,
        0.521450, 0.472806, 0.460935, 0.549113, 0.537101, 0.486201, 0.465843, 0.542203, 0.522633, 0.484192, 0.496887, 0.480551,
        0.463171, 0.555832, 0.459081, 0.494756, 0.584395, 0.575603, 0.505790, 0.568483, 0.497871, 0.464326, 0.464097, 0.507884;

    Matrix<double, 36, 1> x = get_initial_poses();

    std::cout << "initial x: " << std::endl << x << std::endl << std::endl; 

    MatrixXd H(36, 36);
    MatrixXd bT(1, 36);
    MatrixXd b(36, 1);
    MatrixXd dx(36, 1);
    double loss;
    double prev_loss = std::numeric_limits<double>::infinity();
    
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Start pose graph optimization " << std::endl << std::endl; 
    while (true) {
        // clear H and b
        H.setZero();
        b.setZero();

        // clear loss
        loss = 0;

        // calculate the first 11 edges
        for (int i = 0; i <= 10; ++i) {
            int j = i+1;

            Jacobian J_ij = cal_jMatrix(x, Traj.col(i), i, j);
            Matrix3d partial_i = J_ij.i;
            Matrix3d partial_j = J_ij.j;
            Vector3d e_ij = J_ij.e;

            H.block<3, 3>(i*3, i*3) += partial_i.transpose() * partial_i;
            H.block<3, 3>(i*3, j*3) += partial_i.transpose() * partial_j;
            H.block<3, 3>(j*3, i*3) += partial_j.transpose() * partial_i;
            H.block<3, 3>(j*3, j*3) += partial_j.transpose() * partial_j;
            
            b.block<3, 1>(i*3, 0) += partial_i.transpose() * e_ij;
            b.block<3, 1>(j*3, 0) += partial_j.transpose() * e_ij;

            loss += e_ij.transpose() * e_ij;
        }

        // loop closure between the 11-th node and the first node
        int i = 11;
        int j = 0;
        Jacobian J_ij = cal_jMatrix(x, Traj.col(i), i, j);
        Matrix3d partial_i = J_ij.i;
        Matrix3d partial_j = J_ij.j;
        Vector3d e_ij = J_ij.e;

        H.block<3, 3>(i*3, i*3) += partial_i.transpose() * partial_i;
        H.block<3, 3>(i*3, j*3) += partial_i.transpose() * partial_j;
        H.block<3, 3>(j*3, i*3) += partial_j.transpose() * partial_i;
        H.block<3, 3>(j*3, j*3) += partial_j.transpose() * partial_j;

        b.block<3, 1>(i*3, 0) += partial_i.transpose() * e_ij;
        b.block<3, 1>(j*3, 0) += partial_j.transpose() * e_ij;

        // fix the first node
        H.block<3, 3>(0, 0) += MatrixXd::Identity(3, 3);

        // solve dx and update(Using sparse cholesky decomposition)
        SimplicialLLT<SparseMatrix<double>> solver; 

        solver.compute(H.sparseView());
        if (solver.info() != Success) {
            std::cerr << "Cholesky decomposition failed!" << std::endl;
            break;
        }
        dx = solver.solve(-b);
        // VectorXd dx = H.colPivHouseholderQr().solve(-b);

        // PartialPivLU<Eigen::MatrixXd> lu(H);

        // // Solve for x
        // VectorXd dx = lu.solve(-b);

        // LDLT<Eigen::MatrixXd> ldlt(H);

        // // Solve for x
        // VectorXd dx = ldlt.solve(-b);
        x += dx;

        // update loss
        loss += e_ij.transpose() * e_ij;
    
        std::cout << "current loss" << std::endl;
        std::cout << loss << std::endl;

        // when the decrease is less than 1e-9, it is considered to be converged
        if (prev_loss - loss <= 1e-9) {
            std::cout << "Converged!" << std::endl << std::endl;
            break;
        }

        // update prev_loss
        prev_loss = loss;
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "optimized x:" << std::endl;
    std::cout << x << std::endl;
    std::cout << "running duration: " << duration.count() << " seconds" << std::endl;
    return 0;
}

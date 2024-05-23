#include <iostream>
#include <Eigen/Core>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/block_solver.h>

using namespace std; 

typedef Eigen::Matrix<double, 3, 3> Matrix3d;
typedef Eigen::Matrix<double, 3, 1> Vector3d;
typedef std::vector<Matrix3d, Eigen::aligned_allocator<Matrix3d>> PosesVec;

Eigen::MatrixXd Poses(3, 12);

struct Jacobian {
    Matrix3d i;
    Matrix3d j;
    Vector3d e;
    Jacobian(const Matrix3d _i, const Matrix3d _j, const Vector3d _e) : i(_i), j(_j), e(_e) {}
};

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
Jacobian cal_jMatrix(Vector3d v1, Vector3d v2, Vector3d z, int i, int j){
    Matrix3d I = Matrix3d::Zero();
    Matrix3d J = Matrix3d::Zero();
    Vector3d e = Vector3d::Zero();

    Matrix3d Ti = v2t(v1);
    Eigen::Matrix2d Ri = Ti.block<2,2>(0,0);
    Eigen::Vector2d ti = Ti.block<2,1>(0,2);

    Matrix3d Tj = v2t(v2);
    Eigen::Matrix2d Rj = Tj.block<2,2>(0,0);
    Eigen::Vector2d tj = Tj.block<2,1>(0,2);

    Matrix3d T_ij = v2t(z);
    Eigen::Matrix2d R_ij = T_ij.block<2,2>(0,0);
    Eigen::Vector2d t_ij = T_ij.block<2,1>(0,2);
    
    double theta_z = z(2);
    double theta_i = t2v(Ti)(2);
    double theta_j = t2v(Tj)(2);

    Eigen::Matrix2d partial_RiT_theta;
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

/*
Use dead reckoning to find initial poses for graph optimization
Output:
    x: a vector of shape(36, 1) containing 12 initial poses
*/
Eigen::Matrix<double, 36, 1> get_initial_poses(){
    PosesVec poses;
    Matrix3d initial_pose = Matrix3d::Identity(); // initial pose
    poses.push_back(initial_pose);

    // dead reckoning
    // drop the last frame, only 12 frames
    for (size_t i = 0; i < 11; ++i) {
        Vector3d pose = Poses.col(i);
        Matrix3d transformation;
        transformation << std::cos(pose(2)), -std::sin(pose(2)), pose(0),
                          std::sin(pose(2)), std::cos(pose(2)),  pose(1),
                          0,                  0,                   1;
        Matrix3d new_pose = poses.back() * transformation;
        poses.push_back(new_pose);
    }

    Eigen::Matrix<double, 36, 1> x;
    // transform all poses to vector form
    for (size_t i = 0; i < poses.size(); ++i) {
        Eigen::Vector3d v = t2v(poses[i]);
        x(i * 3) = v(0);
        x(i * 3 + 1) = v(1);
        x(i * 3 + 2) = v(2);
    }

    return x;
}


// 定义顶点类型
class VertexPointXYZ : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() {
        _estimate.setZero();
    }

    virtual void oplusImpl(const double* update) {
        _estimate += Eigen::Vector3d(update);
    }

    virtual bool read(std::istream& /*is*/) { return false; }
    virtual bool write(std::ostream& /*os*/) const { return false; }
};

// 定义边类型
class EdgeXYZ : public g2o::BaseBinaryEdge<3, Eigen::Vector3d, VertexPointXYZ, VertexPointXYZ> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void computeError() {
        // std::cout << "123" << std::endl;
        const VertexPointXYZ* v1 = static_cast<const VertexPointXYZ*>(_vertices[0]);
        const VertexPointXYZ* v2 = static_cast<const VertexPointXYZ*>(_vertices[1]);
        int ID_1 = v1->id();
        int ID_2 = v2->id();
        Jacobian J_ij = cal_jMatrix(v1->estimate(), v2->estimate(), _measurement, ID_1, ID_2);
        _error = J_ij.e;
    }

    virtual void linearizeOplus() {
        const VertexPointXYZ* v1 = static_cast<const VertexPointXYZ*>(_vertices[0]);
        const VertexPointXYZ* v2 = static_cast<const VertexPointXYZ*>(_vertices[1]);
        int ID_1 = v1->id();
        int ID_2 = v2->id();
        Jacobian J_ij = cal_jMatrix(v1->estimate(), v2->estimate(), _measurement, ID_1, ID_2);
        _jacobianOplusXi = J_ij.i;
        _jacobianOplusXj = J_ij.j;
    }

    virtual bool read(std::istream& /*is*/) { return false; }
    virtual bool write(std::ostream& /*os*/) const { return false; }
};

int main() {
    Poses << 
        -0.146303, -0.140469, -0.153574, -0.124153, -0.119121, -0.104924, -0.138374, -0.127593, -0.129433, -0.157566, -0.116292, -0.137706,
        0.521450, 0.472806, 0.460935, 0.549113, 0.537101, 0.486201, 0.465843, 0.542203, 0.522633, 0.484192, 0.496887, 0.480551,
        0.463171, 0.555832, 0.459081, 0.494756, 0.584395, 0.575603, 0.505790, 0.568483, 0.497871, 0.464326, 0.464097, 0.507884;

    Eigen::Matrix<double, 36, 1> initial_poses = get_initial_poses();
    std::cout << "initial poses:" << std::endl << initial_poses << std::endl;

    // 创建一个优化器
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 3>> Block;
        
    auto linearSolver = std::make_unique<g2o::LinearSolverDense<Block::PoseMatrixType>>(); // 线性方程求解器
    auto solver_ptr = std::make_unique<Block>(std::move(linearSolver));      // 矩阵块求解器

    g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg(std::move(solver_ptr));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);         // 打开调试输出

    Eigen::Map<Eigen::Matrix<double, 3, 12>> x_m(initial_poses.data());

    // 添加顶点
    const int num_poses = 12; // assuming there are 12 poses
    std::vector<VertexPointXYZ*> vertices(num_poses);

    for (int i = 0; i < num_poses; ++i) {
        VertexPointXYZ* v = new VertexPointXYZ();
        v->setId(i);
        v->setEstimate(x_m.col(i));
        optimizer.addVertex(v);
        vertices[i] = v;
    }
    vertices[0]->setFixed(true);

    // 添加边
    std::vector<EdgeXYZ*> edges(num_poses);
    for (int i = 0; i < num_poses; ++i) {
        EdgeXYZ* edge = new EdgeXYZ();
        edge->setVertex(0, vertices[i]);
        edge->setVertex(1, vertices[(i + 1) % num_poses]); // connecting to the next vertex, cyclic
        edge->setMeasurement(Poses.col(i)); 
        edge->setInformation(Eigen::Matrix3d::Identity()); 
        optimizer.addEdge(edge);
        edges[i] = edge;
    }

    auto start = std::chrono::high_resolution_clock::now();
    // 进行优化
    optimizer.initializeOptimization();
    optimizer.optimize(100); // 迭代优化次数
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "running duration: " << duration.count() << " seconds" << std::endl;

    // 输出优化后的顶点估计值
    for (int i = 0; i < num_poses; ++i) {
        std::cout << vertices[i]->estimate() << std::endl;
    }

    return 0;
}


#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "Thirdparty/g2o/g2o/core/optimizable_graph.h"
#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/g2o/g2o/core/base_binary_edge.h"
#include "Thirdparty/g2o/g2o/core/base_unary_edge.h"
#include "Thirdparty/g2o/g2o/types/se3_ops.h"
// #include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"
// #include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
// #include "Thirdparty/g2o/g2o/solvers/cholmod/linear_solver_cholmod.h"

// 1. 定义 Vertex
namespace GPSFusion {

struct Sim3d {
  double scale = 1;
  Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();

  Sim3d() = default;
  Sim3d(double scale,
        const Eigen::Quaterniond& rotation,
        const Eigen::Vector3d& translation)
      : scale(scale), rotation(rotation), translation(translation) {}

    Sim3d(const Eigen::Matrix<double, 7, 1> & update)
    {

      Eigen::Vector3d omega;
      for (int i=0; i<3; i++)
        omega[i]=update[i];

      Eigen::Vector3d upsilon;
      for (int i=0; i<3; i++)
        upsilon[i]=update[i+3];

      double sigma = update[6];
      double theta = omega.norm();
      Eigen::Matrix3d Omega = g2o::skew(omega);
      scale = std::exp(sigma);
      Eigen::Matrix3d Omega2 = Omega*Omega;
      Eigen::Matrix3d I;
      I.setIdentity();
      Eigen::Matrix3d R;

      double eps = 0.00001;
      double A,B,C;
      if (fabs(sigma)<eps)
      {
        C = 1;
        if (theta<eps)
        {
          A = 1./2.;
          B = 1./6.;
          R = (I + Omega + Omega*Omega);
        }
        else
        {
          double theta2= theta*theta;
          A = (1-cos(theta))/(theta2);
          B = (theta-sin(theta))/(theta2*theta);
          R = I + sin(theta)/theta *Omega + (1-cos(theta))/(theta*theta)*Omega2;
        }
      }
      else
      {
        C=(scale-1)/sigma;
        if (theta<eps)
        {
          double sigma2= sigma*sigma;
          A = ((sigma-1)*scale+1)/sigma2;
          B= ((0.5*sigma2-sigma+1)*scale)/(sigma2*sigma);
          R = (I + Omega + Omega2);
        }
        else
        {
          R = I + sin(theta)/theta *Omega + (1-cos(theta))/(theta*theta)*Omega2;



          double a=scale*sin(theta);
          double b=scale*cos(theta);
          double theta2= theta*theta;
          double sigma2= sigma*sigma;

          double c=theta2+sigma2;
          A = (a*sigma+ (1-b)*theta)/(theta*c);
          B = (C-((b-1)*sigma+a*theta)/(c))*1./(theta2);

        }
      }
      rotation = Eigen::Quaterniond(R);



      Eigen::Matrix3d W = A*Omega + B*Omega2 + C*I;
      translation = W*upsilon;
    }

    Sim3d operator *(const Sim3d& other) const {
      Sim3d ret;
      ret.rotation = rotation*other.rotation;
      ret.translation = scale*(rotation*other.translation)+translation;
      ret.scale = scale*other.scale;
      return ret;
    }

//   static inline Eigen2::Matrix3x4d ToMatrix() {
//     Eigen2::Matrix3x4d matrix;
//     matrix.leftCols<3>() = scale * rotation.toRotationMatrix();
//     matrix.col(3) = translation;
//     return matrix;
//   }

//   static inline Sim3d FromMatrix(const Eigen2::Matrix3x4d& matrix) {
//     Sim3d t;
//     t.scale = matrix.col(0).norm();
//     t.rotation =
//         Eigen::Quaterniond(matrix.leftCols<3>() / t.scale).normalized();
//     t.translation = matrix.rightCols<1>();
//     return t;
//   }
};

class VertexSim3 : public g2o::BaseVertex<7, Sim3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void setToOriginImpl() override {
        _estimate = Sim3d();
    }

    void oplusImpl(const double* update_) override {
        // Eigen::Map<const Eigen::Matrix<double, 7, 1>> update_map(const_cast<double*>(update_));
        // _estimate += update_map;
        Eigen::Map<Eigen::Matrix<double, 7, 1>> update(const_cast<double*>(update_));
        Sim3d s(update);
        setEstimate(s*estimate());
    }

    bool read(std::istream& /*is*/) override {
        return false;
    }

    bool write(std::ostream& /*os*/) const override {
        return false;
    }
};

// 2. 定义 Edge
class EdgeSimilarity : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexSim3> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSimilarity(const Eigen::Vector3d& observed) : g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexSim3>() {
        _observed = observed;
    }

    void computeError() override {
        const VertexSim3* v = static_cast<const VertexSim3*>(_vertices[0]);
        double s = v->estimate().scale;
        Eigen::Vector3d t = v->estimate().translation;
        Eigen::Matrix3d R = v->estimate().rotation.toRotationMatrix();

        Eigen::Vector3d transformed_point = s * R * _observed + t;
        Eigen::Vector3d obs(_measurement);
        _error = transformed_point - obs;
    }

    bool read(std::istream& /*is*/) override {
        return false;
    }

    bool write(std::ostream& /*os*/) const override {
        return false;
    }

    // void linearizeOplus() override {
    //     const VertexSim3* v = static_cast<const VertexSim3*>(_vertices[0]);

    //     double s = v->estimate().scale;
    //     Eigen::Vector3d t = v->estimate().translation;
    //     Eigen::Matrix3d R = v->estimate().rotation.toRotationMatrix();

    //     Eigen::Vector3d xyz_trans = s * R * _observed + t;  // 变换后的点坐标
    //     double x = xyz_trans[0];
    //     double y = xyz_trans[1];
    //     double z = xyz_trans[2];

    //     // 旋转部分的雅可比
    //     _jacobianOplusXi(0, 0) = 0;
    //     _jacobianOplusXi(0, 1) = -z;
    //     _jacobianOplusXi(0, 2) = y;
    //     _jacobianOplusXi(0, 3) = -1;
    //     _jacobianOplusXi(0, 4) = 0;
    //     _jacobianOplusXi(0, 5) = 0;
    //     _jacobianOplusXi(0, 6) = x;  // 尺度部分

    //     _jacobianOplusXi(1, 0) = z;
    //     _jacobianOplusXi(1, 1) = 0;
    //     _jacobianOplusXi(1, 2) = -x;
    //     _jacobianOplusXi(1, 3) = 0;
    //     _jacobianOplusXi(1, 4) = -1;
    //     _jacobianOplusXi(1, 5) = 0;
    //     _jacobianOplusXi(1, 6) = y;  // 尺度部分

    //     _jacobianOplusXi(2, 0) = -y;
    //     _jacobianOplusXi(2, 1) = x;
    //     _jacobianOplusXi(2, 2) = 0;
    //     _jacobianOplusXi(2, 3) = 0;
    //     _jacobianOplusXi(2, 4) = 0;
    //     _jacobianOplusXi(2, 5) = -1;
    //     _jacobianOplusXi(2, 6) = z;  // 尺度部分
    // }
    Eigen::Vector3d _observed;
};
}
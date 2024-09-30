#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>

// struct TError
// {
// 	TError(double t_x, double t_y, double t_z, double xj, double yj, double zj) //残差值
// 				  :t_x(t_x), t_y(t_y), t_z(t_z), xj(xj), yj(yj), zj(zj){}

// 	template <typename T>
// 	bool operator()(const T* w_q_ji, const T* t_ji, T* residuals) const //估计值
// 	{
//         T ti[3];
//         T tj[3];
//         tj[0] = xj;
//         tj[1] = yj;
//         tj[2] = zj;
//         ceres::QuaternionRotatePoint(w_q_ij, tj, ti);
        
// 		residuals[0] = ti[0] + tji[0] - T(t_x);
// 		residuals[1] = ti[1] + tji[0] - T(t_y);
// 		residuals[2] = ti[2] + tji[0] - T(t_z);

// 		return true;
// 	}

// 	static ceres::CostFunction* Create(const double t_x, const double t_y, const double t_z, double xj, double yj, double zj) 
// 	{
// 	  return (new ceres::AutoDiffCostFunction<
// 	          TError, 3, 4, 3>(
// 	          	new TError(t_x, t_y, t_z, xj, yj, zj)));
// 	}

// 	double t_x, t_y, t_z, xj, yj, zj;

// };

class SimilarityCostFunction {
public:
    SimilarityCostFunction(const Eigen::Vector3d& observed, const Eigen::Vector3d& point_source)
        : observed_(observed), point_source_(point_source) {}

    template <typename T>
	bool operator()(const T* s, const T* w_q_ij, const T* tij, T* residuals) const
    {
        // 从参数中提取相似变换矩阵
        T ti[3];
        T tj[3];
        tj[0] = T(point_source_[0]);
        tj[1] = T(point_source_[1]);
        tj[2] = T(point_source_[2]);
        // ceres::QuaternionRotatePoint(w_q_ij, tj, ti);

        ceres::AngleAxisRotatePoint(w_q_ij, tj, ti);
        
		residuals[0] = s[0] * ti[0] + tij[0] - T(observed_[0]);
		residuals[1] = s[0] * ti[1] + tij[1] - T(observed_[1]);
		residuals[2] = s[0] * ti[2] + tij[2] - T(observed_[2]);
        return true;
    }
    // template <typename T>
    // bool operator()(const T* s, const T* w_q_ij, T* residuals) const
    // {
    //     // 从参数中提取相似变换矩阵
    //     T ti[3];
    //     T tj[3];
    //     tj[0] = T(point_source_[0]);
    //     tj[1] = T(point_source_[1]);
    //     tj[2] = T(point_source_[2]);
    //     // ceres::QuaternionRotatePoint(w_q_ij, tj, ti);

    //     ceres::AngleAxisRotatePoint(w_q_ij, tj, ti);
        
	// 	residuals[0] = s[0] * ti[0] + tij[0] - T(observed_[0]);
	// 	residuals[1] = s[0] * ti[1] + tij[1] - T(observed_[1]);
	// 	residuals[2] = s[0] * ti[2] + tij[2] - T(observed_[2]);
    //     return true;
    // }


    static ceres::CostFunction* Create(const Eigen::Vector3d& observed, const Eigen::Vector3d& point_source)
	{
	//   return (new ceres::AutoDiffCostFunction<
	//           SimilarityCostFunction, 3, 1, 4, 3>(
	//           	new SimilarityCostFunction(observed, point_source)));
        return (new ceres::AutoDiffCostFunction<
	          SimilarityCostFunction, 3, 1, 3, 3>(
	          	new SimilarityCostFunction(observed, point_source)));
	}

private:
    Eigen::Vector3d observed_;
    Eigen::Vector3d point_source_;
};
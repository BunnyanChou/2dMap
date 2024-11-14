/******************************************************************************

  This file is part of Map2DFusion.

  Copyright 2016 (c)  Yong Zhao <zd5945@126.com> http://www.zhaoyong.adv-ci.com

  ----------------------------------------------------------------------------

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.

*******************************************************************************/
#include "Map2D.h"
#include "Map2DCPU.h"
#include "Map2DRender.h"
#include "Map2DGPU.h"
#include "MultiBandMap2DCPU.h"
#include "SE3.h"
#include "SO3.h"
#include <iostream>
#include <Eigen/Dense>

using namespace std;

bool Map2DPrepare::prepare(const pi::SE3d& plane, const PinHoleParameters& camera,
                                        const std::deque<std::pair<cv::Mat,pi::SE3d> >& frames)
{
    if(frames.size()==0||camera.w<=0||camera.h<=0||camera.fx==0||camera.fy==0)
    {
        cerr<<"Map2D::prepare:Not valid prepare!\n";
        return false;
    }
    _camera=camera;
    _fxinv=1./camera.fx;
    _fyinv=1./camera.fy;
    _plane =plane;
    _frames=frames;
    for(std::deque<std::pair<cv::Mat,pi::SE3d> >::iterator it=_frames.begin();it!=_frames.end();it++)
    {
        pi::SE3d& pose=it->second;
        std::cout << "r = " << pose.get_rotation() << std::endl;
        std::cout << "t = " << pose.get_translation() << std::endl;
        pose=plane.inverse() * pose;//plane coordinate
        std::cout << "plane_r = " << plane.get_rotation() << std::endl;
        std::cout << "plane_t = " << plane.get_translation() << std::endl;
        std::cout << "r_plane = " << pose.get_rotation() << std::endl;
        std::cout << "t_plane = " << pose.get_translation() << std::endl;
    }

    // plane转到gps坐标系 plane在相机坐标系下的表示为 z = 1
    // pi::Point3d n_w(0., 0., 1.); // 法向量
    // pi::Point3d dn_w(0., 0., 1.); // d

    // pi::Point3d n_gps = Tgpsw.get_rotation() * n_w;
    // pi::Point3d dn_gps = Tgpsw * dn_w;

    // // 平面法向量和平面上点的点积，得到平面参数d_gps
    // double d_gps = n_gps.dot(dn_gps);

    // n_gps = n_gps / n_gps.z;
    // d_gps = d_gps / n_gps.z;
    
    // std::cout << "平面公式：" << n_gps << ", " << d_gps << std::endl;
    // // 把平面参数转化为SE3
    // pi::Point3d n_gps_norm = n_gps.normalize();
    // pi::Point3d t = d_gps * n_gps_norm;

    // pi::Point3d z_axis(0., 0., 1.);
    // pi::Point3d v = z_axis.cross(n_gps_norm);
    // double s = v.norm();
    // double c = z_axis.dot(n_gps_norm);

    // Eigen::Matrix3d vx;
    // vx << 0, -v.z, v.y,
    //       v.z, 0, -v.x,
    //       -v.y, v.x, 0;

    // Eigen::Matrix3d R_matrix;
    // if (s != 0) {
    //     R_matrix = Eigen::Matrix3d::Identity() + vx + vx * vx * ((1 - c) / (s * s));
    // } else {
    //     R_matrix = Eigen::Matrix3d::Identity(); // 若法向量与 z 轴平行，则直接用单位矩阵
    // }

    // std::vector<double> r(R_matrix.data(), R_matrix.data() + R_matrix.size());
    // pi::SO3<double> Rgps(r.data());
    // pi::SE3d Tgps(Rgps, pi::SE3d::Vec3(t.x, t.y, t.z));

    // _plane_gps = Tgps;
    // std::cout << "plane_gps: " << _plane_gps << std::endl;

    // // frames转到gps坐标系
    // _frames_gps=frames;
    // for(std::deque<std::pair<cv::Mat,pi::SE3d> >::iterator it=_frames_gps.begin();it!=_frames_gps.end();it++)
    // {
    //     pi::SE3d& pose=it->second;
    //     pose=_plane_gps.inverse() * Tgpsw * pose;//plane coordinate
    //     std::cout << "pose_gps: " << pose << std::endl;
    // }
    return true;
}

SPtr<Map2D> Map2D::create(int type,bool thread)
{
    if(type==NoType) return SPtr<Map2D>();
    else if(type==TypeCPU)    return SPtr<Map2D>(new Map2DCPU(thread));
    else if(type==TypeMultiBandCPU) return SPtr<MultiBandMap2DCPU>(new MultiBandMap2DCPU(thread));
    else if(type==TypeRender)    return SPtr<Map2D>(new Map2DRender(thread));
    else if(type==TypeGPU)
    {
#ifdef HAS_CUDA
        return SPtr<Map2D>(new Map2DGPU(thread));
#else
        std::cout<<"Warning: CUDA is not enabled, switch to CPU implimentation.\n";
        return SPtr<Map2D>(new Map2DCPU(thread));
#endif
    }
}

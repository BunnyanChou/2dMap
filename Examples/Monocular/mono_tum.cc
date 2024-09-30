/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>

#include "LocalCartesian.hpp"
#include "Map2D.h"
#include "Converter.h"
#include "SE3.h"
using namespace std;

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames,
                vector<double> &vTimestamps);

void InputGPS(const string &strFile, vector<vector<double>> &vGPS, vector<string> &vImageList);

bool obtainFrame(const cv::Mat im, const cv::Mat Twc, std::pair<cv::Mat, pi::SE3d>& frame);

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./mono_tum path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string strFile = string(argv[3])+"/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);

    // load gps
    // string strGPS = string(argv[3]) + "/gps.txt";
    // vector<string> vstrImageList;
    // vector<vector<double>> vGPSs;
    // InputGPS(strGPS, vGPSs, vstrImageList);

    // std::cout << "gps size " << vGPSs.size() << ", " << vGPSs[0][0] << std::endl;
    // if (vstrImageList.size() != vstrImageFilenames.size()) {
    //     cerr << "gps does not match image " << vstrImageList.size() << " v.s " << vstrImageFilenames.size() << std::endl;
    //     return 1;
    // }

    int nImages = vstrImageFilenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::MONOCULAR, true);
    // ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::MONOCULAR_GPS, true);

    // 初始化Map2D
    SPtr<Map2D> map=Map2D::create(Map2D::TypeMultiBandCPU, true); 
    if(!map.get())
    {
        cerr<<"No map2d created!\n";
        return 1;
    }

    // 读内参
    cv::FileStorage fSettings(argv[2], cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    bool map_initialized = false;
    bool pop_first = false;
    deque<std::pair<cv::Mat,pi::SE3d>> frames;
    cv::Mat im;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image from file
        im = cv::imread(string(argv[3])+"/"+vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];
        // vector<double> gps = vGPSs[ni];

        if(im.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenames[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        cv::Mat Tcw = SLAM.TrackMonocular(im, tframe);
        if (!Tcw.empty()) {
            std::cout << "translation: " << Tcw.at<float>(0,3) << Tcw.at<float>(1,3) << Tcw.at<float>(2,3) << std::endl;
        }
        

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        // 图像拼接
        // 前20张图片，加入到frame中去    
        if (!Tcw.empty() && ni < 20) {
            std::pair<cv::Mat,pi::SE3d> frame;
            (void)obtainFrame(im, Tcw, frame);
            frames.push_back(frame);
        }  else if (!Tcw.empty()) {
            if (!map_initialized) {        
                // 把plane, 内参, 初始化frame用于初始化map
                if(!frames.size()) {
                    cout << "Not enough frames. Loaded "<<frames.size()<<" frames.\n";
                    return 1;
                }
                map->prepare(pi::SE3d(), PinHoleParameters(im.cols, im.rows, fx, fy, cx, cy), frames);
                map_initialized = true;
                std::cout << "map_repare done." << std::endl;
            } else {
                // if(map->queueSize()<2) {
                //     // 当小于2张图片时，往里加图片，并且进行拼接
                //     std::pair<cv::Mat,pi::SE3d> frame;
                //     if(!obtainFrame(im, Tcw, frame)) {
                //         break;
                //     }
                //     map->feed(frame.first, frame.second);
                // }
                std::pair<cv::Mat,pi::SE3d> frame;
                (void)obtainFrame(im, Tcw, frame);
                map->feed(frame.first, frame.second);

                // frames.push_back(frame);
                // if (map->queueSize() < 2) {
                //     if (!pop_first) {
                //         int del_num = frames.size() - 2;
                //         for (int idx = 0; idx < del_num; idx++) {
                //             frames.pop_front();
                //         }
                //         pop_first = true;
                //     }
                //     while (!frames.empty()) {
                //         map->feed(frames.front().first, frames.front().second);
                //         frames.pop_front();
                //     }         
                // }
            }
        }
    
        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("/home/heda/zyy/dataset/data_test/fast_stitching/KeyFrameTrajectory.txt");
    SLAM.SaveTrajectoryTUM("/home/heda/zyy/dataset/data_test/fast_stitching/FrameTrajectory.txt");
    SLAM.SaveMap("/home/heda/zyy/dataset/data_test/fast_stitching/map.txt");
    map->save("/home/heda/zyy/dataset/data_test/fast_stitching/map2D.png");
    // Stop all threads
    SLAM.Shutdown();
    return 0;
}

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    string s0;
    // getline(f,s0);
    // getline(f,s0);
    // getline(f,s0);

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
}

void InputGPS(const string &strFile, vector<vector<double>> &vGPS, vector<string> &vImageList)
{
    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    string s0;
    getline(f,s0);
    getline(f,s0);

    GeographicLib::LocalCartesian geoConverter(22.792117, 114.685813, 127.586000);
    bool initGPS = false;
    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            string sRGB;
            ss >> sRGB;
            vImageList.push_back(sRGB);

            double latitude; 
            double longitude; 
            double altitude;
            double posAccuracy;
            ss >> longitude;
            ss >> latitude;
            ss >> altitude;
            ss >> posAccuracy;
            double xyz[3];

            // convert gps from wgs84 to enu
            if(!initGPS)
            {
                geoConverter.Reset(latitude, longitude, altitude);
            }
            geoConverter.Forward(latitude, longitude, altitude, xyz[0], xyz[1], xyz[2]);
            vector<double> tmp{xyz[0], xyz[1], xyz[2], posAccuracy};
            std::cout << sRGB << ", " << latitude << ", "  << longitude << ", " << altitude << std::endl;
            std::cout << "        " << xyz[0] << ", "  << xyz[1] << ", " << xyz[2] << std::endl;
            vGPS.push_back(tmp);
            std::cout << "        " << vGPS[vGPS.size() - 1][0] << ", "  << vGPS[vGPS.size() - 1][1] << ", " << vGPS[vGPS.size() - 1][2] << std::endl;
        }
        initGPS = true;
    }
}

bool obtainFrame(const cv::Mat im, const cv::Mat Tcw, std::pair<cv::Mat, pi::SE3d>& frame)
{
    if (im.empty() || Tcw.empty()) {
        return false;
    }

    cv::Mat mRcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat mRwc = mRcw.t();
    cv::Mat mtcw = Tcw.rowRange(0,3).col(3);
    cv::Mat mtwc = -mRcw.t() * mtcw;

    Eigen::Matrix<double,3,3> eigMat = ORB_SLAM2::Converter::toMatrix3d(mRwc);
    Eigen::Quaterniond r(eigMat);

    frame.first = im;
    frame.second = pi::SE3d(mtwc.at<float>(0), mtwc.at<float>(1), mtwc.at<float>(2), r.x(), r.y(), r.z(), r.w());
    std::cout << mtwc.at<float>(0) << "," << mtwc.at<float>(1) << "," << mtwc.at<float>(2) << "," << r.x() << "," << r.y() << "," << r.z() << "," << r.w() << std::endl;
    return true;
}
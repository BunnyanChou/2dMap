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
#include "Map2DCPU.h"
// #include <gui/gl/glHelper.h>
// #include <GL/gl.h>
// #include <base/Svar/Svar.h>
#include <Global_Timer.h>
// #include <gui/gl/SignalHandle.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include <pangolin/pangolin.h>
// #include "Converter.h"
using namespace std;


/**

  __________max
  |    |    |
  |____|____|
  |    |    |
  |____|____|
 min
 */

bool Map2DCPU::Map2DCPUData::prepare(SPtr<Map2DCPUPrepare> prepared)
{
    if(_w||_h) return false;//already prepared
    // zy prepared是什么时候有的？怎么有的值？怎么更新里面的值？
    {
        _max=pi::Point3d(-1e10,-1e10,-1e10);
        _min=-_max;
        for(std::deque<std::pair<cv::Mat,pi::SE3d> >::iterator it=prepared->_frames.begin();
            it!=prepared->_frames.end();it++)
        {
            pi::SE3d& pose=it->second;
            pi::Point3d& t=pose.get_translation();
            _max.x=t.x>_max.x?t.x:_max.x;
            _max.y=t.y>_max.y?t.y:_max.y;
            _max.z=t.z>_max.z?t.z:_max.z;
            _min.x=t.x<_min.x?t.x:_min.x;
            _min.y=t.y<_min.y?t.y:_min.y;
            _min.z=t.z<_min.z?t.z:_min.z;
        }
        if(_min.z*_max.z<=0) return false;
        cout<<"Box:Min:"<<_min<<",Max:"<<_max<<endl;
    }
    //estimate w,h and bonding box 
    { // intialization _w==0, _h==0的时候， 根据Map2DCPUPrepare获取初值
        double minh;
        if(_min.z>0) minh=_min.z;
        else minh=-_max.z;
        pi::Point3d line=prepared->UnProject(pi::Point2d(prepared->_camera.w,prepared->_camera.h))
                -prepared->UnProject(pi::Point2d(0,0));
        double radius=0.5*minh*sqrt((line.x*line.x+line.y*line.y));
        _lengthPixel=2*radius/sqrt(prepared->_camera.w*prepared->_camera.w
                                   +prepared->_camera.h*prepared->_camera.h);
        // _lengthPixel/=svar.GetDouble("Map2D.Scale",1);
        _lengthPixelInv=1./_lengthPixel;
        _min=_min-pi::Point3d(radius,radius,0);
        _max=_max+pi::Point3d(radius,radius,0);
        pi::Point3d center=0.5*(_min+_max);
        _min=2*_min-center;_max=2*_max-center;
        _eleSize=ELE_PIXELS*_lengthPixel;
        _eleSizeInv=1./_eleSize;
        {
            _w=ceil((_max.x-_min.x)/_eleSize);
            _h=ceil((_max.y-_min.y)/_eleSize);
            _max.x=_min.x+_eleSize*_w;
            _max.y=_min.y+_eleSize*_h;
            _data.resize(_w*_h);
        }
    }
    return true;
}

Map2DCPU::Map2DCPUEle::~Map2DCPUEle()
{
    if(texName) {
        // pi::gl::Signal_Handle::instance().delete_texture(texName);
    }
}

Map2DCPU::Map2DCPU(bool thread)
    :alpha(0/*svar.GetInt("Map2D.Alpha",0)*/),
     _valid(false),_thread(thread)
{
}

bool Map2DCPU::prepare(const pi::SE3d& plane1,const PinHoleParameters& camera,
                const std::deque<std::pair<cv::Mat,pi::SE3d> >& frames)
{
    pi::SE3d plane(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0); // 0.33, -0.038, 0.997, 0.0, 0.0, 0.99998, 0.00609 x, y, z, qx, qy, qz,qw
    //insert frames
    SPtr<Map2DCPUPrepare> p(new Map2DCPUPrepare);
    SPtr<Map2DCPUData>    d(new Map2DCPUData);

    if(p->prepare(plane,camera,frames))
        if(d->prepare(p))
        {
            pi::WriteMutex lock(mutex);
            prepared=p;
            data=d;
            weightImage.release();
            if(_thread&&!isRunning()) {
                start(); 
            }  
            _valid=true;
            return true;
        }
    return false;
}

bool Map2DCPU::feed(cv::Mat img,const pi::SE3d& pose)
{
    if(!_valid) {
        std::cout << "can not run feed" << std::endl; // _valid一直是true,所以不退出
        return false;
    }
    SPtr<Map2DCPUPrepare> p;
    SPtr<Map2DCPUData>    d;
    {
        pi::ReadMutex lock(mutex);
        p=prepared; d=data;
    }
    std::pair<cv::Mat, pi::SE3d> frame(img, p->_plane.inverse() * pose);
    if(_thread) // zy _thread什么时候为false? -- 初始化时确定的, 所以renderFrame一直不执行
    {
        pi::WriteMutex lock(p->mutexFrames);
        p->_frames.push_back(frame);
        if(p->_frames.size()>20) {
            p->_frames.pop_front(); // 元素超过20,就把队首的元素删掉， 为什么？
        }
        return true;
    } else {
        return renderFrame(frame);
    }
}

bool Map2DCPU::renderFrame(const std::pair<cv::Mat,pi::SE3d>& frame)
{
    SPtr<Map2DCPUPrepare> p;
    SPtr<Map2DCPUData>    d;
    {
        pi::ReadMutex lock(mutex);
        p=prepared;d=data;
    }
    if(frame.first.cols!=p->_camera.w||frame.first.rows!=p->_camera.h||frame.first.type()!=CV_8UC3)
    {
        cerr<<"Map2DCPU::renderFrame: frame.first.cols!=p->_camera.w||frame.first.rows!=p->_camera.h||frame.first.type()!=CV_8UC3\n";
        return false;
    }
    // pose->pts
    std::vector<pi::Point2d> imgPts;
    {
        imgPts.reserve(4);
        imgPts.push_back(pi::Point2d(0,0));
        imgPts.push_back(pi::Point2d(p->_camera.w,0));
        imgPts.push_back(pi::Point2d(0,p->_camera.h));
        imgPts.push_back(pi::Point2d(p->_camera.w,p->_camera.h)); // zy 只能有一种大小的图片
    }
    vector<pi::Point2d> pts;
    pts.reserve(imgPts.size());
    pi::Point3d downLook(0,0,-1);
    if(frame.second.get_translation().z < 0) {
        downLook=pi::Point3d(0,0,1);
    }
    for(int i=0;i<imgPts.size();i++)
    {
        pi::Point3d axis=frame.second.get_rotation() * p->UnProject(imgPts[i]); // 四个角点在相机坐标系下的方向
        if(axis.dot(downLook) < 0.4) // 与下视的偏移较大，因此不参与计算
        {
            return false;
        }
        axis = frame.second.get_translation() - axis * (frame.second.get_translation().z / axis.z); // 没有理解
        pts.push_back(pi::Point2d(axis.x, axis.y));
    }
    // dest location?
    double xmin=pts[0].x;
    double xmax=xmin;
    double ymin=pts[0].y;
    double ymax=ymin;
    for(int i=1; i < pts.size(); i++) //找到最大和最小的坐标范围
    {
        if(pts[i].x < xmin) {
            xmin=pts[i].x;
        }
        if(pts[i].y < ymin) {
            ymin=pts[i].y;
        }
        if(pts[i].x > xmax) {
            xmax=pts[i].x;
        }
        if(pts[i].y > ymax) {
            ymax=pts[i].y;
        }
    }
    if(xmin < d->min().x || xmax > d->max().x || ymin < d->min().y || ymax>d->max().y)
    {
        if(p != prepared) {//what if prepare called?
            return false;
        }
        if(!spreadMap(xmin, ymin, xmax, ymax)) {
            return false;
        }
        else
        {
            pi::ReadMutex lock(mutex);
            if(p!=prepared)//what if prepare called?
            {
                return false;
            }
            d = data;//new data
        }
    }
    int xminInt=floor((xmin-d->min().x)*d->eleSizeInv());
    int yminInt=floor((ymin-d->min().y)*d->eleSizeInv());
    int xmaxInt= ceil((xmax-d->min().x)*d->eleSizeInv());
    int ymaxInt= ceil((ymax-d->min().y)*d->eleSizeInv());
    if(xminInt<0||yminInt<0||xmaxInt>d->w()||ymaxInt>d->h()||xminInt>=xmaxInt||yminInt>=ymaxInt)
    {
        cerr<<"Map2DCPU::renderFrame:should never happen!\n";
        return false;
    }
    {
        xmin=d->min().x+d->eleSize()*xminInt;
        ymin=d->min().y+d->eleSize()*yminInt;
        xmax=d->min().x+d->eleSize()*xmaxInt;
        ymax=d->min().y+d->eleSize()*ymaxInt;
    }
    // prepare dst image
    cv::Mat src;
    if(weightImage.empty()||weightImage.cols!=frame.first.cols||weightImage.rows!=frame.first.rows)
    {
        pi::WriteMutex lock(mutex);
        int w = frame.first.cols;
        int h = frame.first.rows;
        weightImage.create(h, w, CV_8UC4);
        pi::byte *p = (weightImage.data);
        float x_center = w/2;
        float y_center = h/2;
        float dis_max = sqrt(x_center * x_center + y_center * y_center);
        int weightType = 0; //svar.GetInt("Map2D.WeightType",0);
        for(int i = 0; i < h; i++)
            for(int j = 0; j < w; j++)
            {
                float dis = (i-y_center)*(i-y_center)+(j-x_center)*(j-x_center);
                dis = 1-sqrt(dis)/dis_max;
                p[1] = p[2] = p[0] = 0;
                if(0==weightType) {
                    p[3]=dis*254.;
                } else {
                    p[3]=dis*dis*254;
                }
                if(p[3]<2) {
                    p[3]=2;
                } 
                p+=4;
            }
        src=weightImage.clone();
    }
    else
    {
        pi::ReadMutex lock(mutex);
        src=weightImage.clone();
    }
    pi::Array_<pi::byte,4> *psrc = (pi::Array_<pi::byte,4>*)src.data;
    pi::Array_<pi::byte,3> *pimg = (pi::Array_<pi::byte,3>*)frame.first.data;
//    float weight=(frame.second.get_rotation()*pi::Point3d(0,0,1)).dot(downLook);
    for(int i=0,iend=weightImage.cols*weightImage.rows;i<iend;i++)
    {
        *((pi::Array_<pi::byte,3>*)psrc)=*pimg;
//        psrc->data[3]*=weight;
        psrc++;
        pimg++;
    }

    if(0/*svar.GetInt("ShowSRC",0)*/)
    {
        cv::imshow("src",src);
    }

    cv::Mat dst((ymaxInt-yminInt) * ELE_PIXELS, (xmaxInt-xminInt) * ELE_PIXELS, src.type());

    std::vector<cv::Point2f> imgPtsCV;
    {
        imgPtsCV.reserve(imgPts.size());
        for(int i=0; i<imgPts.size(); i++)
            imgPtsCV.push_back(cv::Point2f(imgPts[i].x,imgPts[i].y));
    }
    std::vector<cv::Point2f> destPoints;
    destPoints.reserve(imgPtsCV.size());
    for(int i=0;i<imgPtsCV.size();i++)
    {
        destPoints.push_back(cv::Point2f((pts[i].x - xmin) * d->lengthPixelInv(),
                             (pts[i].y - ymin) * d->lengthPixelInv()));
    }

    cv::Mat transmtx = cv::getPerspectiveTransform(imgPtsCV, destPoints); //根据点的关系计算投影矩阵
    cv::warpPerspective(src, dst, transmtx, dst.size(),cv::INTER_LINEAR);

    if(0/*svar.GetInt("ShowDST",0)*/)
    {
        cv::imshow("dst",dst);
    }
    // apply dst to eles
    pi::timer.enter("Apply");
    std::vector<SPtr<Map2DCPUEle> > dataCopy=d->data();
    for(int x=xminInt;x<xmaxInt;x++)
        for(int y=yminInt;y<ymaxInt;y++)
        {
            SPtr<Map2DCPUEle> ele=dataCopy[y*d->w()+x];
            if(!ele.get())
            {
                ele=d->ele(y*d->w()+x);
            }
            {
                pi::WriteMutex lock(ele->mutexData);
                if(ele->img.empty())
                    ele->img=cv::Mat::zeros(ELE_PIXELS,ELE_PIXELS,dst.type());
                pi::Array_<pi::byte,4> *eleP=(pi::Array_<pi::byte,4>*)ele->img.data;
                pi::Array_<pi::byte,4> *dstP=(pi::Array_<pi::byte,4>*)dst.data;
                dstP+=(x-xminInt)*ELE_PIXELS+(y-yminInt)*ELE_PIXELS*dst.cols;
                int skip=dst.cols-ele->img.cols;
                for(int eleY=0;eleY<ELE_PIXELS;eleY++,dstP+=skip)
                    for(int eleX=0;eleX<ELE_PIXELS;eleX++,dstP++,eleP++)
                    {
                        if(eleP->data[3]<dstP->data[3])
                            *eleP=*dstP;
                    }
                ele->Ischanged=true;
            }
        }
    pi::timer.leave("Apply");

    return true;
}


bool Map2DCPU::spreadMap(double xmin,double ymin,double xmax,double ymax)
{
    pi::timer.enter("Map2DCPU::spreadMap");
    SPtr<Map2DCPUData> d;
    {
        pi::ReadMutex lock(mutex);
        d=data;
    }
    int xminInt=floor((xmin-d->min().x)*d->eleSizeInv());
    int yminInt=floor((ymin-d->min().y)*d->eleSizeInv());
    int xmaxInt= ceil((xmax-d->min().x)*d->eleSizeInv());
    int ymaxInt= ceil((ymax-d->min().y)*d->eleSizeInv());
    xminInt=min(xminInt, 0); 
    yminInt=min(yminInt, 0);
    xmaxInt=max(xmaxInt, d->w()); 
    ymaxInt=max(ymaxInt, d->h());
    int w = xmaxInt - xminInt;
    int h = ymaxInt - yminInt;
    pi::Point2d min, max;
    {
        min.x = d->min().x+d->eleSize()*xminInt;
        min.y = d->min().y+d->eleSize()*yminInt;
        max.x = min.x+w*d->eleSize();
        max.y = min.y+h*d->eleSize();
    }
    std::vector<SPtr<Map2DCPUEle>> dataOld = d->data();
    std::vector<SPtr<Map2DCPUEle>> dataCopy; // 大小为什么是和w, h一致的？
    dataCopy.resize(w*h);
    {
        for(int x=0, xend=d->w(); x<xend; x++)
            for(int y=0,yend=d->h(); y<yend; y++)
            {
                dataCopy[x-xminInt + (y-yminInt) * w] = dataOld[x + y * d->w()]; //zy 初始化是在哪里初始化的？
            }
    }
    //apply
    {
        pi::WriteMutex lock(mutex);
        data=SPtr<Map2DCPUData>(new Map2DCPUData(d->eleSize(),d->lengthPixel(),
                                                 pi::Point3d(max.x, max.y, d->max().z),
                                                 pi::Point3d(min.x, min.y, d->min().z),
                                                 w, h, xminInt, yminInt, dataCopy));
    }
    pi::timer.leave("Map2DCPU::spreadMap");
    return true;
}

bool Map2DCPU::drawFrame(cv::Mat &stitchImage)
{
    if (!_valid) return false;
    SPtr<Map2DCPUPrepare> p;
    SPtr<Map2DCPUData>    d;
    {
        pi::ReadMutex lock(mutex);
        p=prepared;
        d=data;
    }

    // cv::Mat stitchImagetest;
    if(stitchImage.empty()) {
        stitchImage.create(d->h()*ELE_PIXELS,d->w()*ELE_PIXELS,CV_8UC4);
    } else if (stitchImage.cols != d->w()*ELE_PIXELS || stitchImage.rows != d->h()*ELE_PIXELS) { //地图发生了扩张
        // 扩张之前图像old和现在的图像new之间的对应关系 扩张时dataCopy[x-xminInt + (y-yminInt) * w] = dataOld[x + y * d->w()];
        // 求xminInt和yminInt
        cv::Mat spreadMapmat(d->h()*ELE_PIXELS, d->w()*ELE_PIXELS, CV_8UC4);
        int xminInt = d->minIntx();
        int yminInt = d->minInty();
        int w = stitchImage.cols;
        int h = stitchImage.rows;
        int newW = min((xminInt+d->w())*ELE_PIXELS, w);
        int newH = min((yminInt+d->h())*ELE_PIXELS, h);
        stitchImage.copyTo(spreadMapmat(cv::Rect(-xminInt*ELE_PIXELS,-yminInt*ELE_PIXELS,newW,newH)));
        stitchImage = spreadMapmat.clone();
    }
    std::vector<SPtr<Map2DCPUEle>> dataCopy=d->data();
    int wCopy=d->w(),hCopy=d->h();
    for(int x=0;x<wCopy;x++)
    {
        for(int y=0;y<hCopy;y++)
        {
            int idxData=y*wCopy+x;
            float x0=d->min().x+x*d->eleSize();
            float y0=d->min().y+y*d->eleSize();
            float x1=x0+d->eleSize();
            float y1=y0+d->eleSize();
            SPtr<Map2DCPUEle> ele=dataCopy[idxData];
            if(!ele.get())  continue;
            if(ele->img.empty()) continue;
            // if(ele->texName==0)
            // {
            //     glGenTextures(1, &ele->texName);
            // }
            if(ele->Ischanged)
            {
                pi::ReadMutex lock1(ele->mutexData);
                ele->img.copyTo(stitchImage(cv::Rect(ELE_PIXELS*(x),ELE_PIXELS*(y),ELE_PIXELS,ELE_PIXELS)));
                ele->Ischanged=false;
            }
        }
    }
    cv::Mat img_resized;
    cv::resize(stitchImage, img_resized, cv::Size(stitchImage.cols/8, stitchImage.rows/8), 0, 0, cv::INTER_AREA);
    cv::imshow("Map2DCPU Visualization", img_resized);

    cv::waitKey(30);  // 确保图像刷新
    return true;
}

bool Map2DCPU::getFrame(std::pair<cv::Mat,pi::SE3d>& frame)
{
    // 获取队首元素
    pi::ReadMutex lock(mutex);
    pi::ReadMutex lock1(prepared->mutexFrames);
    if(prepared->_frames.size()) {
        frame=prepared->_frames.front();
        prepared->_frames.pop_front();
        return true;
    } else {
        return false;
    }
}

void Map2DCPU::run()
{
    std::pair<cv::Mat,pi::SE3d> frame;
    while(!shouldStop())
    {
        if(_valid)
        {
            if(getFrame(frame))
            {
                pi::timer.enter("Map2DCPU::renderFrame");
                renderFrame(frame);
                pi::timer.leave("Map2DCPU::renderFrame");
            }
        }
        sleep(10);
    }
}

// void Map2DCPU::draw()
// {
//     if(!_valid) return;

//     SPtr<Map2DCPUPrepare> p;
//     SPtr<Map2DCPUData>    d;
//     {
//         pi::ReadMutex lock(mutex);
//         p=prepared;d=data;
//     }
//     glMatrixMode(GL_MODELVIEW);
//     glPushMatrix();

//     cv::Mat planeMat = cv::Mat::eye(4,4,CV_64F);
//     double r[12];
//     p->_plane.getMatrix(r);
//     for(int i=0;i<3;i++)
//     {
//         for(int j=0;j<4;j++)
//         {
//             planeMat.at<double>(i,j)=r[i*4+j];
//         }
//     }
//     glMultMatrixd(planeMat.ptr<GLdouble>(0));
//     //draw deque frames
//     pi::TicTac ticTac;
//     ticTac.Tic();
//     {
//         std::deque<std::pair<cv::Mat,pi::SE3d> > frames=p->getFrames();
//         glDisable(GL_LIGHTING);
//         glBegin(GL_LINES);
//         for(std::deque<std::pair<cv::Mat,pi::SE3d> >::iterator it=frames.begin();it!=frames.end();it++)
//         {
//             pi::SE3d& pose=it->second;
//             pi::Point3d trans = pose.get_translation();
//             pi::Point3d axis_x = pose*pi::Point3d(1,0,0);
//             pi::Point3d axis_y = pose*pi::Point3d(0,1,0);
//             pi::Point3d axis_z = pose*pi::Point3d(0,0,1);
//             glColor3ub(255,0,0);
//             glVertex3d(trans.x, trans.y, trans.z);
//             glVertex3d(axis_x.x, axis_x.y, axis_x.z);
//             glColor3ub(0,255,0);
//             glVertex3d(trans.x, trans.y, trans.z);
//             glVertex3d(axis_y.x, axis_y.y, axis_y.z);
//             glColor3ub(0,0,255);
//             glVertex3d(trans.x, trans.y, trans.z);
//             glVertex3d(axis_z.x, axis_z.y, axis_z.z);
//         }
//         glEnd();
//     }
//     //draw global area
//     {
//         pi::Point3d _min=d->min();
//         pi::Point3d _max=d->max();
//         glColor3ub(255,0,0);
//         glBegin(GL_LINES);
//         glVertex3d(_min.x,_min.y,0);
//         glVertex3d(_min.x,_max.y,0);
//         glVertex3d(_min.x,_min.y,0);
//         glVertex3d(_max.x,_min.y,0);
//         glVertex3d(_max.x,_min.y,0);
//         glVertex3d(_max.x,_max.y,0);
//         glVertex3d(_min.x,_max.y,0);
//         glVertex3d(_max.x,_max.y,0);
//         glEnd();
//     }

//     //draw textures
//     glEnable(GL_TEXTURE_2D);
//     glEnable(GL_BLEND);
// //    glEnable(GL_LIGHTING);
//     if(alpha)
//     {
//         glEnable(GL_ALPHA_TEST);
//         glAlphaFunc(GL_GREATER, 0.1f);
//         glBlendFunc(GL_SRC_ALPHA,GL_ONE);
//     }
//     GLint last_texture_ID;
//     glGetIntegerv(GL_TEXTURE_BINDING_2D, &last_texture_ID);
//     std::vector<SPtr<Map2DCPUEle> > dataCopy=d->data();
//     int wCopy=d->w(),hCopy=d->h();
//     glColor3ub(255,255,255);
//     for(int x=0;x<wCopy;x++)
//         for(int y=0;y<hCopy;y++)
//         {
//             int idxData=y*wCopy+x;
//             float x0=d->min().x+x*d->eleSize();
//             float y0=d->min().y+y*d->eleSize();
//             float x1=x0+d->eleSize();
//             float y1=y0+d->eleSize();
//             SPtr<Map2DCPUEle> ele=dataCopy[idxData];
//             if(!ele.get())  continue;
//             if(ele->img.empty()) continue;
//             if(ele->texName==0)
//             {
//                 glGenTextures(1, &ele->texName);
//             }
//             if(ele->Ischanged&&ticTac.Tac()<0.02)
//             {
//                 pi::timer.enter("glTexImage2D");
//                 pi::ReadMutex lock1(ele->mutexData);
//                 glBindTexture(GL_TEXTURE_2D,ele->texName);
// //                if(ele->img.elemSize()==1)
//                     glTexImage2D(GL_TEXTURE_2D, 0,
//                                  GL_RGBA, ele->img.cols,ele->img.rows, 0,
//                                  GL_BGRA, GL_UNSIGNED_BYTE,ele->img.data);
//                 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,  GL_LINEAR);
//                 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_NEAREST);
//                 ele->Ischanged=false;
//                 pi::timer.leave("glTexImage2D");
//             }
//             glBindTexture(GL_TEXTURE_2D,ele->texName);
//             glBegin(GL_QUADS);
//             glTexCoord2f(0.0f, 0.0f); glVertex3f(x0,y0,0);
//             glTexCoord2f(0.0f, 1.0f); glVertex3f(x0,y1,0);
//             glTexCoord2f(1.0f, 1.0f); glVertex3f(x1,y1,0);
//             glTexCoord2f(1.0f, 0.0f); glVertex3f(x1,y0,0);
//             glEnd();
//         }
//     glBindTexture(GL_TEXTURE_2D, last_texture_ID);
//     glPopMatrix();
// }

bool Map2DCPU::save(const std::string& filename)
{
    // determin minmax
    SPtr<Map2DCPUPrepare> p;
    SPtr<Map2DCPUData>    d;
    {
        pi::ReadMutex lock(mutex);
        p=prepared;d=data;
    }
    if(d->w()==0||d->h()==0) return false;

    pi::Point2i minInt(1e6,1e6),maxInt(-1e6,-1e6);
    for(int x=0;x<d->w();x++)
        for(int y=0;y<d->h();y++)
        {
            SPtr<Map2DCPUEle> ele=d->data()[x+y*d->w()];
            if(!ele.get()) continue;
            {
                pi::ReadMutex lock(ele->mutexData);
                if(ele->img.empty()) continue;
            }
            minInt.x=min(minInt.x,x); minInt.y=min(minInt.y,y);
            maxInt.x=max(maxInt.x,x); maxInt.y=max(maxInt.y,y);
        }

    maxInt=maxInt+pi::Point2i(1,1);
    pi::Point2i wh=maxInt-minInt;
    cv::Mat result(wh.y*ELE_PIXELS,wh.x*ELE_PIXELS,CV_8UC4);
    for(int x=minInt.x;x<maxInt.x;x++)
        for(int y=minInt.y;y<maxInt.y;y++)
        {
            SPtr<Map2DCPUEle> ele=d->data()[x+y*d->w()];
            if(!ele.get()) continue;
            {
                pi::ReadMutex lock(ele->mutexData);
                ele->img.copyTo(result(cv::Rect(ELE_PIXELS*(x-minInt.x),ELE_PIXELS*(y-minInt.y),ELE_PIXELS,ELE_PIXELS)));
            }
        }

    cv::imwrite(filename, result);
    return true;
}

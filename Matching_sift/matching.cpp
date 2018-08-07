#include <fstream>
#include <iostream>
#include <string>
#include <omp.h>
#include <boost/filesystem.hpp>
#include <flann/flann.hpp>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>
#include <opencv/cvaux.h>
#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include "DataUtility.h"
#include "StructDefinition.h"

int GetStaticCorrespondences(vector<Point> x1, vector<Point> x2, vector<bool> &vIsInlier);
void Matching(vector<FrameCamera> &vFC, int currentFC, CvMat *K, CvMat *invK, double omega, vector<Feature> &feature_static);
using namespace std;

void Undistortion(CvMat *K, CvMat *invK, double omega, vector<double> &vx,  vector<double> &vy)
{
        for (int iPoint = 0; iPoint < vx.size(); iPoint++)
        {
                CvMat *x_homo = cvCreateMat(3,1,CV_32FC1);
                cvSetReal2D(x_homo, 0, 0, vx[iPoint]);
                cvSetReal2D(x_homo, 1, 0, vy[iPoint]);
                cvSetReal2D(x_homo, 2, 0, 1);
                CvMat *x_homo_n = cvCreateMat(3,1,CV_32FC1);
                cvMatMul(invK, x_homo, x_homo_n);
                double x_n, y_n;
                x_n = cvGetReal2D(x_homo_n, 0, 0);
                y_n = cvGetReal2D(x_homo_n, 1, 0);
                double r_d = sqrt(x_n*x_n+y_n*y_n);
                double r_u = tan(r_d*omega)/2/tan(omega/2);
                double x_u = r_u/r_d*x_n;
                double y_u = r_u/r_d*y_n;
                CvMat *x_undist_n = cvCreateMat(3,1,CV_32FC1);
                cvSetReal2D(x_undist_n, 0, 0, x_u);
                cvSetReal2D(x_undist_n, 1, 0, y_u);
                cvSetReal2D(x_undist_n, 2, 0, 1);
                CvMat *x_undist = cvCreateMat(3,1,CV_32FC1);
                cvMatMul(K, x_undist_n, x_undist);
                vx[iPoint] = cvGetReal2D(x_undist,0,0);
                vy[iPoint] = cvGetReal2D(x_undist,1,0);

                cvReleaseMat(&x_homo);
                cvReleaseMat(&x_homo_n);
                cvReleaseMat(&x_undist_n);
                cvReleaseMat(&x_undist);
        }
}

int main ( int argc, char * argv[] )
{	
	string path = "";	
	string savepath = path + "reconstruction/";
	string savepath_m = savepath + "measurement/";
	boost::filesystem::create_directories(savepath.c_str());
	boost::filesystem::create_directories(savepath_m.c_str());

	string filelist = path + "image/filelist.list";
	string savefile_static = savepath_m + "static_measurement_desc%07d.txt";

	FileName fn;	
	vector<Camera> vCamera;
		
	vector<string> vFilename;
	LoadFileListData(filelist, vFilename);

	Camera cam;
	cam.id = 0;
	for (int iFile = 0; iFile < vFilename.size(); iFile++)
	{		
		cam.vImageFileName.push_back(path + "image/" +vFilename[iFile]);
		cam.vTakenFrame.push_back(iFile);		
	}
	vCamera.push_back(cam);

	ifstream fin_cal;
	string calibfile = path + "image/calib_fisheye.txt";
	fin_cal.open(calibfile.c_str(), ifstream::in);
	string dummy;
	int im_width, im_height;
	double focal_x, focal_y, princ_x, princ_y, omega;
	double distCtrX, distCtrY;
	fin_cal >> dummy >> im_width;
	fin_cal >> dummy >> im_height;
	fin_cal >> dummy >> focal_x;
	fin_cal >> dummy >> focal_y;
	fin_cal >> dummy >> princ_x;
	fin_cal >> dummy >> princ_y;
	fin_cal >> dummy >> omega;

	CvMat *K = cvCreateMat(3,3,CV_32FC1);
	cvSetIdentity(K);
	cvSetReal2D(K, 0, 0, focal_x);
	cvSetReal2D(K, 0, 2, princ_x);
	cvSetReal2D(K, 1, 1, focal_y);
	cvSetReal2D(K, 1, 2, princ_y);
	CvMat *invK = cvCreateMat(3,3,CV_32FC1);
	cvInvert(K, invK);
	fin_cal.close();

	vector<int> vFrameOrder;
	if (vCamera[0].vTakenFrame.size()%2==0)
	{
		for (int i = 0; i < vCamera[0].vTakenFrame.size()/2; i++)
		{
			vFrameOrder.push_back(i);
			vFrameOrder.push_back(vCamera[0].vTakenFrame.size()-i-1);
		}
	}
	else
	{
		for (int i = 0; i < (vCamera[0].vTakenFrame.size()-1)/2; i++)
		{
			vFrameOrder.push_back(i);
			vFrameOrder.push_back(vCamera[0].vTakenFrame.size()-i-1);
		}
		vFrameOrder.push_back((vCamera[0].vTakenFrame.size()-1)/2);
	}

	vector<int> vCameraID;
	vector<string> vImageFileName;
	vector<FrameCamera> vFC;

	vector<vector<SIFT_Descriptor> > vvSift_desc;
	vFC.resize(vCamera[0].vTakenFrame.size());
	for (int iFrame = 0; iFrame < vCamera[0].vTakenFrame.size(); iFrame++)
	{
		FrameCamera fc;
		string keyFile = vCamera[0].vImageFileName[iFrame];
		fc.imageFileName = vCamera[0].vImageFileName[iFrame];	
		fc.imageFileName[fc.imageFileName.size()-3] = 'b';
		fc.imageFileName[fc.imageFileName.size()-2] = 'm';
		fc.imageFileName[fc.imageFileName.size()-1] = 'p';

		fc.cameraID = 0;
		fc.frameIdx = iFrame;

		vector<SIFT_Descriptor> vSift_desc;
		LoadSIFTData_int(keyFile, vSift_desc);

		vector<double> vx1, vy1;
		vector<double> dis_vx1, dis_vy1;		

		for (int isift = 0; isift < vSift_desc.size(); isift++)
		{				
			vx1.push_back(vSift_desc[isift].x);
			vy1.push_back(vSift_desc[isift].y);
		}
		Undistortion(K, invK, omega, vx1,  vy1);
		for (int isift = 0; isift < vSift_desc.size(); isift++)
		{
			vSift_desc[isift].x = vx1[isift];
			vSift_desc[isift].y = vy1[isift];
		}

		fc.vSift_desc = vSift_desc;
		vSift_desc.clear();
		vFC[iFrame] = fc;
	}

	int nTotal = vFrameOrder.size();
	int current = 0;
	vector<int> vFrameIdx;
	vFrameIdx.resize(vFrameOrder.size(), 0);
	
	#pragma omp parallel for 
	for (int iFC = 0; iFC < vFrameOrder.size(); iFC++)
	{
		int iFC1 = vFrameOrder[iFC];
		FrameCamera cFC = vFC[iFC1];
		vector<Feature> feature_static;
		Matching(vFC, iFC1, K, invK, omega, feature_static);
			
		for (int iFeature = 0; iFeature < feature_static.size(); iFeature++)
		{
			feature_static[iFeature].id = 0;
		}
		char temp[1000];
		sprintf(temp, savefile_static.c_str(), iFC1);
		string savefile_static1 = temp;
		SaveMeasurementData_RGB_DESC(savefile_static1, feature_static, FILESAVE_WRITE_MODE);
		feature_static.clear();

		vFrameIdx[iFC] = 1;
		int count = 0;
		for (int ic = 0; ic < vFrameIdx.size(); ic++)
		{
			if (vFrameIdx[ic] == 1)
				count++;
		}

		cout << "Status: " << count << " " << nTotal << endl;
	}
	return 0;
} // end main()

void Matching(vector<FrameCamera> &vFC, int currentFC, CvMat *K, CvMat *invK, double omega, vector<Feature> &feature_static)
{
	if (vFC[currentFC].vSift_desc.size() == 0)
		return;
 
	FrameCamera cFC = vFC[currentFC];	
	flann::Matrix<float> descM1(new float[cFC.vSift_desc.size()*128], cFC.vSift_desc.size(), 128);
	for (int iDesc = 0; iDesc < cFC.vSift_desc.size(); iDesc++)
	{
		for (int iDim = 0; iDim < 128; iDim++)
		{
			descM1[iDesc][iDim] = (float) cFC.vSift_desc[iDesc].vDesc[iDim];
		}
	}
	flann::Index<flann::L2<float> > index1(descM1, flann::KDTreeIndexParams(4));
	index1.buildIndex();

	IplImage *iplImg1 = cvLoadImage(cFC.imageFileName.c_str());
	for (int iFeature = 0; iFeature < cFC.vSift_desc.size(); iFeature++)
	{
		Feature fs;
		fs.vCamera.push_back(cFC.cameraID);
		fs.vFrame.push_back(cFC.frameIdx);
		fs.vx.push_back(cFC.vSift_desc[iFeature].x);
		fs.vy.push_back(cFC.vSift_desc[iFeature].y);
		fs.vx_dis.push_back(cFC.vSift_desc[iFeature].dis_x);
		fs.vy_dis.push_back(cFC.vSift_desc[iFeature].dis_y);
		fs.vvDesc.push_back(cFC.vSift_desc[iFeature].vDesc);
		CvScalar s;
		s = cvGet2D(iplImg1,((int)cFC.vSift_desc[iFeature].dis_y),((int)cFC.vSift_desc[iFeature].dis_x));
		fs.b = s.val[0];
		fs.g = s.val[1];
		fs.r = s.val[2];
		feature_static.push_back(fs);
	}
	cvReleaseImage(&iplImg1);
	
	// Matching to the rest images
	vector<Point> featureSequence;
	for (int iSecondFrame = currentFC+1; iSecondFrame < vFC.size(); iSecondFrame++)
	{
		vector<int> vIdx1, vIdx2;
		int nn = 2;
		int nPoint1 = cFC.vSift_desc.size();
		int nPoint2 = vFC[iSecondFrame].vSift_desc.size();
		if (nPoint2<20)
			continue;
		
		flann::Matrix<int> result12(new int[nPoint1*nn], nPoint1, nn);
		flann::Matrix<float> dist12(new float[nPoint1*nn], nPoint1, nn);

		flann::Matrix<int> result21(new int[nPoint2*nn], nPoint2, nn);
		flann::Matrix<float> dist21(new float[nPoint2*nn], nPoint2, nn);

		flann::Matrix<float> descM2(new float[vFC[iSecondFrame].vSift_desc.size()*128], vFC[iSecondFrame].vSift_desc.size(), 128);
		for (int iDesc = 0; iDesc < vFC[iSecondFrame].vSift_desc.size(); iDesc++)
		{
			for (int iDim = 0; iDim < 128; iDim++)
			{
				descM2[iDesc][iDim] = (float) vFC[iSecondFrame].vSift_desc[iDesc].vDesc[iDim];
			}
		}

		flann::Index<flann::L2<float> > index2(descM2, flann::KDTreeIndexParams(4));
		index2.buildIndex();

		index2.knnSearch(descM1, result12, dist12, nn, flann::SearchParams(128));
		index1.knnSearch(descM2, result21, dist21, nn, flann::SearchParams(128));
		delete[] descM2.ptr();
		
		for (int iFeature = 0; iFeature < nPoint1; iFeature++)
		{
			float dist1 = dist12[iFeature][0];
			float dist2 = dist12[iFeature][1];

			if (dist1/dist2 < 0.7)
			{
				int idx12 = result12[iFeature][0];

				dist1 = dist21[idx12][0];
				dist2 = dist21[idx12][1];

				if (dist1/dist2 < 0.7)
				{
					int idx21 = result21[idx12][0];
	
					if (iFeature == idx21)
					{					
						Point p1, p2;
						p1.x = cFC.vSift_desc[idx21].x;
						p1.y = cFC.vSift_desc[idx21].y;

						p2.x = vFC[iSecondFrame].vSift_desc[idx12].x;
						p2.y = vFC[iSecondFrame].vSift_desc[idx12].y;
						
						double dist = sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y));
						if (dist < 5e-0)
							continue;
					
						vIdx1.push_back(idx21);
						vIdx2.push_back(idx12);
					}
				}

				int idx21 = result21[idx12][0];
				if (iFeature == idx21)
				{
					vIdx1.push_back(idx21);
					vIdx2.push_back(idx12);
				}

				vIdx1.push_back(iFeature);
				vIdx2.push_back(idx12);
			}
		}

		delete[] result12.ptr();
		delete[] result21.ptr();
		delete[] dist12.ptr();
		delete[] dist21.ptr();

		vector<Point> x1, x2;
		for (int iIdx = 0; iIdx < vIdx1.size(); iIdx++)
		{
			Point p1, p2;
			p1.x = cFC.vSift_desc[vIdx1[iIdx]].x;
			p1.y = cFC.vSift_desc[vIdx1[iIdx]].y;

			p2.x = vFC[iSecondFrame].vSift_desc[vIdx2[iIdx]].x;
			p2.y = vFC[iSecondFrame].vSift_desc[vIdx2[iIdx]].y;

			x1.push_back(p1);
			x2.push_back(p2);
		}

		if (x1.size() < 20)
		{
			continue;
		}
		vector<bool> vIsInlier;
		if (GetStaticCorrespondences(x1, x2, vIsInlier) < 20)
		{
			continue;
		}

		vector<int> vTempIdx1, vTempIdx2;
		for (int iIsInlier = 0; iIsInlier < vIsInlier.size(); iIsInlier++)
		{
			if (vIsInlier[iIsInlier])
			{
				if (vTempIdx1.size() > 0)
				{
					if ((vTempIdx1[vTempIdx1.size()-1] == vIdx1[iIsInlier]) && (vTempIdx2[vTempIdx2.size()-1] == vIdx2[iIsInlier]))
					{
						continue;
					}
				}
				vTempIdx1.push_back(vIdx1[iIsInlier]);
				vTempIdx2.push_back(vIdx2[iIsInlier]);
			}
		}
		vIdx1 = vTempIdx1;
		vIdx2 = vTempIdx2;

		if (vIdx1.size() < 20)
			continue;

		for (int iInlier = 0; iInlier < vIdx2.size(); iInlier++)
		{
			feature_static[vIdx1[iInlier]].vCamera.push_back(vFC[iSecondFrame].cameraID);
			feature_static[vIdx1[iInlier]].vFrame.push_back(vFC[iSecondFrame].frameIdx);
			feature_static[vIdx1[iInlier]].vx.push_back(vFC[iSecondFrame].vSift_desc[vIdx2[iInlier]].x);
			feature_static[vIdx1[iInlier]].vx_dis.push_back(vFC[iSecondFrame].vSift_desc[vIdx2[iInlier]].dis_x);
			feature_static[vIdx1[iInlier]].vy.push_back(vFC[iSecondFrame].vSift_desc[vIdx2[iInlier]].y);
			feature_static[vIdx1[iInlier]].vy_dis.push_back(vFC[iSecondFrame].vSift_desc[vIdx2[iInlier]].dis_y);
			feature_static[vIdx1[iInlier]].vvDesc.push_back(vFC[iSecondFrame].vSift_desc[vIdx2[iInlier]].vDesc);
		}			
		cout << "Frame 1 : " << cFC.cameraID << " " << cFC.frameIdx << "  Frame 2 : " << vFC[iSecondFrame].cameraID << " " << vFC[iSecondFrame].frameIdx << " /" << vIdx1.size() << endl;
	}

	vector<Feature> vTempFeature;
	for (int iFeature = 0; iFeature < feature_static.size(); iFeature++)
	{
		if (feature_static[iFeature].vCamera.size() > 1)
		{
			vTempFeature.push_back(feature_static[iFeature]);
		}
	}
	feature_static.clear();
	feature_static = vTempFeature;
	vTempFeature.clear();

	delete[] descM1.ptr();
}
int GetStaticCorrespondences(vector<Point> x1, vector<Point> x2, vector<bool> &vIsInlier)
{
	vector<int> vInlierID;
	CvMat *cx1 = cvCreateMat(x1.size(), 2, CV_32FC1);
	CvMat *cx2 = cvCreateMat(x1.size(), 2, CV_32FC1);
	for (int ix = 0; ix < x1.size(); ix++)
	{
		cvSetReal2D(cx1, ix, 0, x1[ix].x);
		cvSetReal2D(cx1, ix, 1, x1[ix].y);
		cvSetReal2D(cx2, ix, 0, x2[ix].x);
		cvSetReal2D(cx2, ix, 1, x2[ix].y);
	}

	vector<cv::Point2f> points1(x1.size());
	vector<cv::Point2f> points2(x2.size());
	for (int ip = 0; ip < x1.size(); ip++)
	{
		cv::Point2f p1, p2;
		p1.x = x1[ip].x;
		p1.y = x1[ip].y;
		p2.x = x2[ip].x;
		p2.y = x2[ip].y;
		points1[ip] = p1;
		points2[ip] = p2;
	}

	CvMat *status = cvCreateMat(1,cx1->rows,CV_8UC1);
	CvMat *F = cvCreateMat(3,3,CV_32FC1);
	int n = cvFindFundamentalMat(cx1, cx2, F, CV_FM_LMEDS, 1, 0.99, status);
	//cv::Mat FundamentalMatrix = cv::findFundamentalMat(points1, points2, cv::FM_LMEDS, 3, 0.99);
	if (n != 1)
	{
		cvReleaseMat(&status);
		cvReleaseMat(&F);
		cvReleaseMat(&cx1);
		cvReleaseMat(&cx2);
		return 0;
	}
	int nP=0;
	double ave = 0;
	int nInliers = 0;
	for (int i = 0; i < cx1->rows; i++)
	{
		if (cvGetReal2D(status, 0, i) == 1)
		{
			//vIsInlier.push_back(true);
			nP++;
			CvMat *xM2 = cvCreateMat(1,3,CV_32FC1);
			CvMat *xM1 = cvCreateMat(3,1,CV_32FC1);
			CvMat *s = cvCreateMat(1,1, CV_32FC1);
			cvSetReal2D(xM2, 0, 0, x2[i].x);
			cvSetReal2D(xM2, 0, 1, x2[i].y);
			cvSetReal2D(xM2, 0, 2, 1);
			cvSetReal2D(xM1, 0, 0, x1[i].x);
			cvSetReal2D(xM1, 1, 0, x1[i].y);
			cvSetReal2D(xM1, 2, 0, 1);
			cvMatMul(xM2, F, xM2);
			cvMatMul(xM2, xM1, s);			

			double l1 = cvGetReal2D(xM2, 0, 0);
			double l2 = cvGetReal2D(xM2, 0, 1);
			double l3 = cvGetReal2D(xM2, 0, 2);

			double dist = abs(cvGetReal2D(s, 0, 0))/sqrt(l1*l1+l2*l2);

			if (dist < 5)
			{
				vIsInlier.push_back(true);
				nInliers++;
			}
			else
			{
				vIsInlier.push_back(false);
			}
			ave += dist;

			cvReleaseMat(&xM2);
			cvReleaseMat(&xM1);
			cvReleaseMat(&s);
		}
		else
		{
			vIsInlier.push_back(false);
		}
	}
	//cout << ave/nP << endl;

	cvReleaseMat(&status);
	cvReleaseMat(&F);
	cvReleaseMat(&cx1);
	cvReleaseMat(&cx2);
	if (ave/nP > 10)
		return 0;
	return nInliers;
}

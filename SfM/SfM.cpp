// StaticReconstruction.cpp : Defines the entry point for the console application.
//
#include <cv.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <time.h>
#include <omp.h>
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include "CeresUtility.h"
#include "DataUtility.h"
#include "epnp.h"
#include "MathUtility.h"
#include "MultiviewUtility.h"

#define CAMERA_REG_RANSAC_THRESHOLD 10e-0
#define CAMERA_REG_RANSAC_MAX_ITER 1e+4
#define MAX_FRAMES 1e+5

using namespace std;
int main ( int argc, char * argv[] )
{
	srand(time(NULL));
	//////////////////////////////////////////////////////////////////////////////////////////////
	// Load data
	string initialFile = "reconstruction/initial_frame.txt";
	string measurementFile = "reconstruction/stitchedmeasurement_static.txt";
	string cameraFile = "reconstruction/camera.txt";
	string structureFile = "reconstruction/structure.txt";

	vector<Feature> vFeature;
	vector<string> vFilename;
	int max_nFrames;

	///////////////////////////////////////////////
	// Load initial pair file
	int initial_frame1, initial_frame2, intial_nFrames;
	LoadInitialFileData(initialFile, initial_frame1, initial_frame2, intial_nFrames);

	///////////////////////////////////////////////
	// Load calibration file
	string calibfile = "image/calib_fisheye.txt";
	CvMat *K_data = cvCreateMat(3, 3, CV_32FC1);
	double omega;
	LoadCalibrationData(calibfile, K_data, omega);

	///////////////////////////////////////////////
	// Load camera structure
	Camera camera;
	camera.id = 0;
	camera.nFrames = intial_nFrames;
	camera.K = K_data;
	for (int iFrame = 0; iFrame < camera.nFrames; iFrame++)
	{
		camera.vFrame.push_back(iFrame);
		camera.vTakenFrame.push_back(iFrame);
		camera.vTakenInstant.push_back(iFrame);
	}

	///////////////////////////////////////////////
	// Load correspondence file
	LoadCorrespondenceData(measurementFile, vFeature);
	max_nFrames = camera.nFrames;
	
	///////////////////////////////////////////////
	// Associating frame with visible features 
	camera.vvFeatureID.resize(max_nFrames);
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		for (int iFrame = 0; iFrame < vFeature[iFeature].vFrame.size(); iFrame++)
		{
			camera.vvFeatureID[vFeature[iFeature].vFrame[iFrame]].push_back(iFeature);
		}
	}	
	
	///////////////////////////////////////////////
	// Shuffling the frames for reconstruction
	vector<int> vFrameOrder;
	int frame1, frame2;
	frame1 = initial_frame1; frame2 = initial_frame2;
	vector<int> vTempFrame1 = camera.vFrame;
	while(!vTempFrame1.empty())
	{
		int randk = rand()%vTempFrame1.size();
		vFrameOrder.push_back(vTempFrame1[randk]);
		vTempFrame1.erase(vTempFrame1.begin()+randk);
	}

	///////////////////////////////////////////////
	// Camera pose estimation using F matrix between frame1 and frame2
	CvMat *P = cvCreateMat(3, 4, CV_32FC1);
	CvMat *X = cvCreateMat(vFeature.size(), 3, CV_32FC1);
	cvSetZero(X);
	CvMat *P0 = cvCreateMat(3, 4, CV_32FC1);	cvSetIdentity(P0);
	cvMatMul(camera.K, P0, P0);
	vector<CvMat *> cP;
	vector<int> vUsedFrame;
	cout << "Two first frames: " << frame1 << " " << frame2 << endl;
	int nStr = CameraPoseEstimation(vFeature, frame1, frame2, camera, P, X);
	cout << "Number of features to do bilinear camera pose estimation: " << nStr << endl;
	cP.push_back(P0);
	cP.push_back(cvCloneMat(P));
	vUsedFrame.push_back(frame1);
	vUsedFrame.push_back(frame2);

	// Bundle adjustment
	CeresSolverGoPro(vFeature, vUsedFrame, cP, X, camera.K,	omega);

	// Save result
	SaveCameraData(cameraFile, cP, camera.K, vUsedFrame);
	SaveStructureData_RGB_fast(structureFile, X, vFeature);

	///////////////////////////////////////////////
	// Incremental bundle adjustment
	PrintAlgorithm("Incremental Bundle Adjustment");
	int iterF = 0;
	bool isFailed = false;
	vector<int> failedFrame;
	vector<double> vnInlier;
	///////////////////////////////////////////////
	// vFrameOrder: a list of un-reconstructed frames
	while(vFrameOrder.size() != 0)
	{
		///////////////////////////////////////////////
		// Find the best next image (frame) to reconstruct
		// The one that has the maximum number of 2D-3D correspondences
		int cFrame, tempFrame = 0;
		int max_intersection = 0;
		int max_intersection_frame = -1;
		cout << "Reconstructed frames: " << cP.size() << " out of " << max_nFrames << endl;
		///////////////////////////////////////////////
		// Push the failed attempted frames back to the un-reconstructed frame list if the previous frame was successful
		if (!isFailed)
		{
			for (int iFrame = 0; iFrame < failedFrame.size(); iFrame++)
			{
				vFrameOrder.push_back(failedFrame[iFrame]);
			}
		}
		///////////////////////////////////////////////
		// Find 2D-3D correspondences among reconstructed (registered) features
		vector<int> vIntersection, vIntersection_frame;
		vIntersection.resize(vFrameOrder.size(),0);
		vIntersection_frame.resize(vFrameOrder.size());
		bool isOn = false;
		
		vector<Feature> vFeature_temp_regi;
		for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
		{
			if (vFeature[iFeature].isRegistered)
				vFeature_temp_regi.push_back(vFeature[iFeature]);
		}
		
		#pragma omp parallel for
		for (int itemp = 0; itemp < vFrameOrder.size(); itemp++)
		{
			vIntersection[itemp] = 0;
			vIntersection_frame[itemp] = -1;
			if (isOn) // if frames with enough correspondences (>600) are found, skip the rest
				continue;
			int cFrame1 = vFrameOrder[itemp];
			vector<int>::iterator it = find(vUsedFrame.begin(),vUsedFrame.end(),cFrame1);
			if (it != vUsedFrame.end())
				continue;

			int nVisible = GetFeatureGivenFrame(vFeature_temp_regi, cFrame1);
			vIntersection[itemp] = nVisible;
			vIntersection_frame[itemp] = cFrame1;
			if (nVisible > 600)
				isOn = true;
		}

		///////////////////////////////////////////////
		// Find the best one
		for (int itemp = 0; itemp < vIntersection.size(); itemp++)
		{
			if (vIntersection[itemp]>max_intersection)
			{
				max_intersection = vIntersection[itemp];
				max_intersection_frame = vIntersection_frame[itemp];
			}
			if (max_intersection > 600)
				break;
		}
		cFrame = max_intersection_frame;
		iterF++;

		if ((vFrameOrder.size() > MAX_FRAMES) || (max_intersection_frame == -1) || (max_intersection <= 40))
		{
			cout << "Total Reconstructed cameras: ";
			for (int i = 0; i < vUsedFrame.size(); i++)
				cout << vUsedFrame[i] << " ";
			cout << endl;
			break;
		}
		///////////////////////////////////////////////
		// Reconstruct cFrame
		cout << endl <<  "-------------" << endl;
		vector<int>::const_iterator it = find(camera.vTakenFrame.begin(), camera.vTakenFrame.end(), cFrame);
		if (it == camera.vTakenFrame.end())
		{
			return 0;
		}
		int iTakenFrame = (int) (it - camera.vTakenFrame.begin());
		cout << "Processing " << cFrame << "th frame ..." << endl;
		///////////////////////////////////////////////
		// Remove cFrame from the vFrameOrder
		vector<int> visibleID;
		vector<int> tempFrameOrder;
		for (int iFrameOrder = 0; iFrameOrder < vFrameOrder.size(); iFrameOrder++)
		{
			if (vFrameOrder[iFrameOrder] != cFrame)
				tempFrameOrder.push_back(vFrameOrder[iFrameOrder]);
		}
		vFrameOrder = tempFrameOrder;

		vector<vector<double> > cx_vec, cX_vec;
		if (!Get2D3DCorrespondence(vFeature, cFrame, X, cx_vec, cX_vec, visibleID))
		{
			failedFrame.push_back(cFrame);
			cout << "Not enough 2D-3D correpondences" << endl;
			continue;
		}

		///////////////////////////////////////////////
		// Register camera using ePnP RANSAC
		cout << "The number of 2D-3D correspondences: " << visibleID.size() << endl;
		CvMat *cx = cvCreateMat(cx_vec.size(), cx_vec[0].size(), CV_32FC1);
		CvMat *cX = cvCreateMat(cX_vec.size(), cX_vec[0].size(), CV_32FC1);
		SetCvMatFromVectors(cx_vec, cx);
		SetCvMatFromVectors(cX_vec, cX);
		cx_vec.clear();
		cX_vec.clear();
		CvMat *P1 = cvCreateMat(3,4,CV_32FC1);
		vector<int> vInlierRANSAC;
		if (!CameraRegistration(cX, cx, camera.K, P1, CAMERA_REG_RANSAC_THRESHOLD, CAMERA_REG_RANSAC_MAX_ITER, vInlierRANSAC))
		{
			cout << "Not enough inliers for ePnP" << endl;
			isFailed = true;
			failedFrame.push_back(cFrame);
			cvReleaseMat(&cx);
			cvReleaseMat(&cX);
			cvReleaseMat(&P1);
			continue;
		}
		else
		{
			isFailed = false;
			failedFrame.clear();
		}

		cvReleaseMat(&cx);
		cvReleaseMat(&cX);
		vUsedFrame.push_back(cFrame);
		cP.push_back(P1);

		///////////////////////////////////////////////
		// Add 3D points by triangulating with the reconstructed frames
		vector<CvMat *> v_newX3;
		vector<vector<int> > v_vIdx;
		v_newX3.resize(vUsedFrame.size()-1);
		v_vIdx.resize(vUsedFrame.size()-1);
		for (int iUsedFrame = 0; iUsedFrame < vUsedFrame.size()-1; iUsedFrame++)
		{
			///////////////////////////////////////////////
			// Find 2D-2D correspondences which are un-reconstruected features	
			vector<vector<double> > cx1_vec, cx2_vec;
			vector<int> visibleID1;
			if (!GetUnreconstructedCorrespondences(vFeature, cFrame, vUsedFrame[iUsedFrame], camera.vvFeatureID[iTakenFrame], cx1_vec, cx2_vec, visibleID1))
				continue;

			vector<vector<double> > ex1_vec, ex2_vec;
			vector<int> eVisibleID;
			ex1_vec = cx1_vec;
			ex2_vec = cx2_vec;
			eVisibleID = visibleID1;

			CvMat *ex1 = cvCreateMat(ex1_vec.size(), ex1_vec[0].size(), CV_32FC1);
			CvMat *ex2 = cvCreateMat(ex2_vec.size(), ex2_vec[0].size(), CV_32FC1);
			SetCvMatFromVectors(ex1_vec, ex1);
			SetCvMatFromVectors(ex2_vec, ex2);
			ex1_vec.clear();
			ex2_vec.clear();
			///////////////////////////////////////////////
			// Linear triangulation
			vector<int> filteredFeatureIDforTriangulation;
			vector<vector<double> > newX_vec;
			if (!LinearTriangulation(ex1, P1, ex2, cP[iUsedFrame], eVisibleID, newX_vec, filteredFeatureIDforTriangulation))
			{
				cvReleaseMat(&ex1);
				cvReleaseMat(&ex2);
				continue;
			}

			eVisibleID = filteredFeatureIDforTriangulation;
			CvMat *newX = cvCreateMat(newX_vec.size(), newX_vec[0].size(), CV_32FC1);
			SetCvMatFromVectors(newX_vec, newX);
			cvReleaseMat(&ex1);
			cvReleaseMat(&ex2);
			newX_vec.clear();

			///////////////////////////////////////////////
			// Reject the points behind the cameras
			vector<int> eVisibleID1;
			vector<vector<double> > newX1_vec;
			if (!ExcludePointBehindCamera(newX, P1, cP[iUsedFrame], eVisibleID, eVisibleID1, newX1_vec))
			{
				cvReleaseMat(&newX);
				continue;
			}
			cvReleaseMat(&newX);			
			CvMat *newX1 = cvCreateMat(newX1_vec.size(), newX1_vec[0].size(), CV_32FC1);
			SetCvMatFromVectors(newX1_vec, newX1);
			newX1_vec.clear();
			///////////////////////////////////////////////
			// Reject points at infinity
			vector<int>::const_iterator it = find(camera.vTakenFrame.begin(), camera.vTakenFrame.end(), vUsedFrame[iUsedFrame]);
			int idx1 = (int)(it - camera.vTakenFrame.begin());
			vector<vector<double> > newX2_vec;
			vector<int> eVisibleID2;
			if (!ExcludePointAtInfinity(newX1, P1, cP[iUsedFrame], camera.K, eVisibleID1, eVisibleID2, newX2_vec))
			{
				cvReleaseMat(&newX1);
				continue;
			}
			cvReleaseMat(&newX1);	
			CvMat *newX2 = cvCreateMat(newX2_vec.size(), newX2_vec[0].size(), CV_32FC1);
			SetCvMatFromVectors(newX2_vec, newX2);
			newX2_vec.clear();
			///////////////////////////////////////////////
			// Reject points with high reprojection error
			vector<int> eVisibleID3;
			vector<vector<double> > newX3_vec;
			if (!ExcludePointHighReprojectionError(vFeature, cP, vUsedFrame, eVisibleID2, newX2, eVisibleID3, newX3_vec, omega, camera.K))
			{
				cvReleaseMat(&newX2);
				continue;
			}	
			cvReleaseMat(&newX2);
			v_newX3[iUsedFrame] = cvCreateMat(newX3_vec.size(), newX3_vec[0].size(), CV_32FC1);
			v_vIdx[iUsedFrame]=eVisibleID3;
			SetCvMatFromVectors(newX3_vec, v_newX3[iUsedFrame]);
			newX3_vec.clear();
		}

		///////////////////////////////////////////////
		// Set reconstructed points
		for (int iUsedFrame = 0; iUsedFrame < v_vIdx.size(); iUsedFrame++)
		{
			if (v_vIdx[iUsedFrame].size() == 0)
				continue;
			cout << "Number of features added from " << vUsedFrame[iUsedFrame] << "th frame: " << v_vIdx[iUsedFrame].size() << endl;
			SetIndexedMatRowwise(X, v_vIdx[iUsedFrame], v_newX3[iUsedFrame]);

			for (int iVisibleId = 0; iVisibleId < v_vIdx[iUsedFrame].size(); iVisibleId++)
			{
				vFeature[v_vIdx[iUsedFrame][iVisibleId]].isRegistered = true;
			}
			cvReleaseMat(&v_newX3[iUsedFrame]);
		}

		if ((int)cP.size() < 100)
		{
			int nStructure = ExcludePointHighReprojectionError(vFeature, cP, vUsedFrame, X, omega, camera.K);
			cout << "Number of reconstructed frames: " << cP.size() << endl;
			cout << "Number of reconstructed structure: " << nStructure << endl;;
			CeresSolverGoPro(vFeature, vUsedFrame, cP, X, camera.K, omega);
		}
		else if ((int)cP.size() < 200)
		{
			if ((int)cP.size() % 2 == 0)
			{
				int nStructure = ExcludePointHighReprojectionError(vFeature, cP, vUsedFrame, X, omega, camera.K);
				cout << "Number of reconstructed frames: " << cP.size() << endl;
				cout << "Number of reconstructed structure: " << nStructure << endl;;
				CeresSolverGoPro(vFeature, vUsedFrame, cP, X, camera.K, omega);
			}
		}
		else if ((int) cP.size() < 600)
		{
			if ((int)cP.size() % 3 == 0)
			{
				int nStructure = ExcludePointHighReprojectionError(vFeature, cP, vUsedFrame, X, omega, camera.K);
				cout << "Number of reconstructed frames: " << cP.size() << endl;
				cout << "Number of reconstructed structure: " << nStructure << endl;;
				CeresSolverGoPro(vFeature, vUsedFrame, cP, X, camera.K, omega);
			}
		}
		else if ((int) cP.size() < 1000)
		{
			if ((int)cP.size() % 8 == 0)
			{
				int nStructure = ExcludePointHighReprojectionError(vFeature, cP, vUsedFrame, X, omega, camera.K);
				cout << "Number of reconstructed frames: " << cP.size() << endl;
				cout << "Number of reconstructed structure: " << nStructure << endl;;
				CeresSolverGoPro(vFeature, vUsedFrame, cP, X, camera.K, omega);
			}
		}
		else 
		{
			if ((int)cP.size() % 12 == 0)
			{
				int nStructure = ExcludePointHighReprojectionError(vFeature, cP, vUsedFrame, X, omega, camera.K);
				cout << "Number of reconstructed frames: " << cP.size() << endl;
				cout << "Number of reconstructed structure: " << nStructure << endl;;
				CeresSolverGoPro(vFeature, vUsedFrame, cP, X, camera.K, omega);
			}
		}
		SaveCameraData(cameraFile, cP, camera.K, vUsedFrame);
		SaveStructureData_RGB_fast(structureFile, X, vFeature);
	}

	int nStructure = ExcludePointHighReprojectionError(vFeature, cP, vUsedFrame, X, omega, camera.K);
	cout << "Number of reconstructed frames: " << cP.size() << endl;
	cout << "Number of reconstructed structure: " << nStructure << endl;;
	CeresSolverGoPro(vFeature, vUsedFrame, cP, X, camera.K, omega);
	SaveCameraData(cameraFile, cP, camera.K, vUsedFrame);
	SaveStructureData_RGB_fast(structureFile, X, vFeature);

	return 0;
}

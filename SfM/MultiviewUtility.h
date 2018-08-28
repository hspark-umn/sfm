#ifndef MULTIVIEWUTILITY_H
#define MULTIVIEWUTILITY_H
#include <cv.h>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <cmath>
#include "StructDefinition.h"
#include "MathUtility.h"
#include "epnp.h"
using namespace std;
#define POINT_AT_INFINITY_ZERO 1e-2

int CameraRegistration(CvMat *X, CvMat *x, CvMat *K, CvMat *P, double ransacThreshold, int ransacMaxIter, vector<int> &vInlierOut);
void GetCorrespondence(vector<Feature> &vFeature, int frame1, int frame2, vector<vector<double> > &cx1, vector<vector<double> > &cx2, vector<int> &featureID);
void GetExtrinsicParameterFromE(CvMat *E, CvMat *x1, CvMat *x2, CvMat *P);
int CameraPoseEstimation(vector<Feature> &vFeature, int initialFrame1, int initialFrame2, Camera camera, CvMat *P, CvMat *X);
int Get2D3DCorrespondence(vector<Feature> &vFeature, int frame1, CvMat *X, vector<vector<double> > &cx, vector<vector<double> > &cX, vector<int> &visibleID);
int EPNP_ExtrinsicCameraParamEstimation(CvMat *X, CvMat *x, CvMat *K, CvMat *P);
int GetUnreconstructedCorrespondences(vector<Feature> &vFeature, int frame1, int frame2, vector<int> vFeatureID, vector<vector<double> > &cx1, vector<vector<double> > &cx2, vector<int> &visibleID);
int LinearTriangulation(CvMat *x1, CvMat *P1, CvMat *x2, CvMat *P2, vector<int> &featureID, vector<vector<double> > &X, vector<int> &filteredFeatureID);
void LinearTriangulation(CvMat *x1, CvMat *P1, CvMat *x2, CvMat *P2, CvMat *X);
int ExcludePointBehindCamera(CvMat *X, CvMat *P1, CvMat *P2, vector<int> &featureID, vector<int> &excludedFeatureID, vector<vector<double> > &cX);
int ExcludePointAtInfinity(CvMat *X, CvMat *P1, CvMat *P2, CvMat *K, vector<int> &featureID, vector<int> &excludedFeatureID, vector<vector<double> > &cX);
bool ExcludePointHighReprojectionError(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> &vUsedFrame
	, vector<int> &visibleStrucrtureID, CvMat *X_tot
	, vector<int> &visibleStrucrtureID_new, vector<vector<double> > &X_tot_new,
	double omega, CvMat *K);
int ExcludePointHighReprojectionError(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> vUsedFrame, CvMat *X_tot, double omega, CvMat *K);
void SetCvMatFromVectors(vector<vector<double> > x, CvMat *X);
void GetExtrinsicParameterFromE(CvMat *E, CvMat *x1, CvMat *x2, CvMat *P);
int GetFeatureGivenFrame(vector<Feature> &vFeature, int frame1);


#endif //MULTIVIEWUTILITY_H
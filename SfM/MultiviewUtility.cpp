#include "MultiviewUtility.h"
#include <cmath>
#include <math.h>
#include <algorithm>
using namespace std;

void GetCorrespondence(vector<Feature> &vFeature, int frame1, int frame2, vector<vector<double> > &cx1, vector<vector<double> > &cx2, vector<int> &featureID)
{
	vector<double> x1, y1, x2, y2;
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		vector<int>::iterator it1 = find(vFeature[iFeature].vFrame.begin(), vFeature[iFeature].vFrame.end(), frame1);
		vector<int>::iterator it2 = find(vFeature[iFeature].vFrame.begin(), vFeature[iFeature].vFrame.end(), frame2);

		if ((it1 != vFeature[iFeature].vFrame.end()) && (it2 != vFeature[iFeature].vFrame.end()))
		{
			int idx = int(it1 - vFeature[iFeature].vFrame.begin());
			x1.push_back(vFeature[iFeature].vx[idx]);
			y1.push_back(vFeature[iFeature].vy[idx]);
			idx = int(it2 - vFeature[iFeature].vFrame.begin());
			x2.push_back(vFeature[iFeature].vx[idx]);
			y2.push_back(vFeature[iFeature].vy[idx]);
			featureID.push_back(vFeature[iFeature].id);
		}
	}
	for (int i = 0; i < x1.size(); i++)
	{
		vector<double> x1_vec, x2_vec;
		x1_vec.push_back(x1[i]);
		x1_vec.push_back(y1[i]);

		x2_vec.push_back(x2[i]);
		x2_vec.push_back(y2[i]);

		cx1.push_back(x1_vec);
		cx2.push_back(x2_vec);
	}
}

void GetExtrinsicParameterFromE(CvMat *E, CvMat *x1, CvMat *x2, CvMat *P)
{
	CvMat *W = cvCreateMat(3, 3, CV_32FC1);
	CvMat *U = cvCreateMat(3, 3, CV_32FC1);
	CvMat *D = cvCreateMat(3, 3, CV_32FC1);
	CvMat *Vt = cvCreateMat(3, 3, CV_32FC1);
	CvMat *Wt = cvCreateMat(3, 3, CV_32FC1);
	cvSVD(E, D, U, Vt, CV_SVD_V_T);

	cvSetReal2D(W, 0, 0, 0);	cvSetReal2D(W, 0, 1, -1);	cvSetReal2D(W, 0, 2, 0);
	cvSetReal2D(W, 1, 0, 1);	cvSetReal2D(W, 1, 1, 0);	cvSetReal2D(W, 1, 2, 0);
	cvSetReal2D(W, 2, 0, 0);	cvSetReal2D(W, 2, 1, 0);	cvSetReal2D(W, 2, 2, 1);
	cvTranspose(W, Wt);

	CvMat *P0 = cvCreateMat(3, 4, CV_32FC1);
	cvSetIdentity(P0);

	CvMat *P1 = cvCreateMat(3, 4, CV_32FC1);
	CvMat *P2 = cvCreateMat(3, 4, CV_32FC1);
	CvMat *P3 = cvCreateMat(3, 4, CV_32FC1);
	CvMat *P4 = cvCreateMat(3, 4, CV_32FC1);

	CvMat *R1 = cvCreateMat(3, 3, CV_32FC1);
	CvMat *R2 = cvCreateMat(3, 3, CV_32FC1);
	CvMat *t1 = cvCreateMat(3, 1, CV_32FC1);
	CvMat *t2 = cvCreateMat(3, 1, CV_32FC1);
	CvMat *temp33 = cvCreateMat(3, 3, CV_32FC1);

	cvMatMul(U, W, temp33);
	cvMatMul(temp33, Vt, R1);
	cvMatMul(U, Wt, temp33);
	cvMatMul(temp33, Vt, R2);

	cvSetReal2D(t1, 0, 0, cvGetReal2D(U, 0, 2));
	cvSetReal2D(t1, 1, 0, cvGetReal2D(U, 1, 2));
	cvSetReal2D(t1, 2, 0, cvGetReal2D(U, 2, 2));
	ScalarMul(t1, -1, t2);

	SetSubMat(P1, 0, 0, R1);
	SetSubMat(P1, 0, 3, t1);
	SetSubMat(P2, 0, 0, R1);
	SetSubMat(P2, 0, 3, t2);
	SetSubMat(P3, 0, 0, R2);
	SetSubMat(P3, 0, 3, t1);
	SetSubMat(P4, 0, 0, R2);
	SetSubMat(P4, 0, 3, t2);
	if (cvDet(R1) < 0)
	{
		ScalarMul(P1, -1, P1);
		ScalarMul(P2, -1, P2);
	}

	if (cvDet(R2) < 0)
	{
		ScalarMul(P3, -1, P3);
		ScalarMul(P4, -1, P4);
	}
	CvMat *X1 = cvCreateMat(x1->rows, 3, CV_32FC1);
	LinearTriangulation(x1, P0, x2, P1, X1);
	CvMat *X2 = cvCreateMat(x1->rows, 3, CV_32FC1);;
	LinearTriangulation(x1, P0, x2, P2, X2);
	CvMat *X3 = cvCreateMat(x1->rows, 3, CV_32FC1);;
	LinearTriangulation(x1, P0, x2, P3, X3);
	CvMat *X4 = cvCreateMat(x1->rows, 3, CV_32FC1);;
	LinearTriangulation(x1, P0, x2, P4, X4);

	int x1neg = 0, x2neg = 0, x3neg = 0, x4neg = 0;
	CvMat *H1 = cvCreateMat(4, 4, CV_32FC1);	CvMat *invH1 = cvCreateMat(4, 4, CV_32FC1);		CvMat *HX1 = cvCreateMat(X1->rows, X1->cols, CV_32FC1);
	cvSetIdentity(H1);
	SetSubMat(H1, 0, 0, P1);
	cvInvert(H1, invH1);
	Pxx_inhomo(H1, X1, HX1);

	CvMat *H2 = cvCreateMat(4, 4, CV_32FC1);	CvMat *invH2 = cvCreateMat(4, 4, CV_32FC1);		CvMat *HX2 = cvCreateMat(X1->rows, X1->cols, CV_32FC1);
	cvSetIdentity(H2);
	SetSubMat(H2, 0, 0, P2);
	cvInvert(H2, invH2);
	Pxx_inhomo(H2, X2, HX2);
	CvMat *H3 = cvCreateMat(4, 4, CV_32FC1);	CvMat *invH3 = cvCreateMat(4, 4, CV_32FC1);		CvMat *HX3 = cvCreateMat(X1->rows, X1->cols, CV_32FC1);
	cvSetIdentity(H3);
	SetSubMat(H3, 0, 0, P3);
	cvInvert(H3, invH3);
	Pxx_inhomo(H3, X3, HX3);
	CvMat *H4 = cvCreateMat(4, 4, CV_32FC1);	CvMat *invH4 = cvCreateMat(4, 4, CV_32FC1);		CvMat *HX4 = cvCreateMat(X1->rows, X1->cols, CV_32FC1);
	cvSetIdentity(H4);
	SetSubMat(H4, 0, 0, P4);
	cvInvert(H4, invH4);
	Pxx_inhomo(H4, X4, HX4);

	for (int ix = 0; ix < x1->rows; ix++)
	{
		if ((cvGetReal2D(X1, ix, 2)<0) || (cvGetReal2D(HX1, ix, 2)<0))
			x1neg++;
		if ((cvGetReal2D(X2, ix, 2)<0) || (cvGetReal2D(HX2, ix, 2)<0))
			x2neg++;
		if ((cvGetReal2D(X3, ix, 2)<0) || (cvGetReal2D(HX3, ix, 2)<0))
			x3neg++;
		if ((cvGetReal2D(X4, ix, 2)<0) || (cvGetReal2D(HX4, ix, 2)<0))
			x4neg++;
	}

	CvMat *temp34 = cvCreateMat(3, 4, CV_32FC1);
	if ((x1neg <= x2neg) && (x1neg <= x3neg) && (x1neg <= x4neg))
		SetSubMat(P, 0, 0, P1);
	else if ((x2neg <= x1neg) && (x2neg <= x3neg) && (x2neg <= x4neg))
		SetSubMat(P, 0, 0, P2);
	else if ((x3neg <= x1neg) && (x3neg <= x2neg) && (x3neg <= x4neg))
		SetSubMat(P, 0, 0, P3);
	else
		SetSubMat(P, 0, 0, P4);

	//cout << x1neg << " " << x2neg << " " << " " << x3neg << " " << x4neg << endl;
	cvReleaseMat(&W);
	cvReleaseMat(&U);
	cvReleaseMat(&D);
	cvReleaseMat(&Vt);
	cvReleaseMat(&Wt);
	cvReleaseMat(&P0);
	cvReleaseMat(&P1);
	cvReleaseMat(&P2);
	cvReleaseMat(&P3);
	cvReleaseMat(&P4);
	cvReleaseMat(&R1);
	cvReleaseMat(&R2);
	cvReleaseMat(&t1);
	cvReleaseMat(&t2);
	cvReleaseMat(&temp33);
	cvReleaseMat(&temp34);
	cvReleaseMat(&H1);
	cvReleaseMat(&invH1);
	cvReleaseMat(&H2);
	cvReleaseMat(&invH2);
	cvReleaseMat(&H3);
	cvReleaseMat(&invH3);
	cvReleaseMat(&H4);
	cvReleaseMat(&invH4);
	cvReleaseMat(&X1);
	cvReleaseMat(&X2);
	cvReleaseMat(&X3);
	cvReleaseMat(&X4);

	cvReleaseMat(&HX1);
	cvReleaseMat(&HX2);
	cvReleaseMat(&HX3);
	cvReleaseMat(&HX4);
}

int Get2D3DCorrespondence(vector<Feature> &vFeature, int frame1, CvMat *X, vector<vector<double> > &cx, vector<vector<double> > &cX, vector<int> &visibleID)
{
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		if (!vFeature[iFeature].isRegistered)
			continue;
		vector<double> cx_vec, cX_vec;
		vector<int>::iterator it1 = find(vFeature[iFeature].vFrame.begin(), vFeature[iFeature].vFrame.end(), frame1);
		if (it1 != vFeature[iFeature].vFrame.end())
		{
			int idx1 = int(it1 - vFeature[iFeature].vFrame.begin());

			cx_vec.push_back(vFeature[iFeature].vx[idx1]);
			cx_vec.push_back(vFeature[iFeature].vy[idx1]);

			cX_vec.push_back(cvGetReal2D(X, vFeature[iFeature].id, 0));
			cX_vec.push_back(cvGetReal2D(X, vFeature[iFeature].id, 1));
			cX_vec.push_back(cvGetReal2D(X, vFeature[iFeature].id, 2));

			cx.push_back(cx_vec);
			cX.push_back(cX_vec);

			visibleID.push_back(vFeature[iFeature].id);
		}
	}

	if (cx.size() < 1)
		return 0;
	return visibleID.size();
}



int CameraPoseEstimation(vector<Feature> &vFeature, int initialFrame1, int initialFrame2, Camera camera, CvMat *P, CvMat *X)
{
	PrintAlgorithm("Camera Pose Estimation");
	vector<int> visibleFeatureID;
	vector<vector<double> > cx1_vec, cx2_vec, nx1_vec, nx2_vec;

	GetCorrespondence(vFeature, initialFrame1, initialFrame2, cx1_vec, cx2_vec, visibleFeatureID);
	CvMat *cx1 = cvCreateMat(cx1_vec.size(), 2, CV_32FC1);
	CvMat *cx2 = cvCreateMat(cx2_vec.size(), 2, CV_32FC1);
	SetCvMatFromVectors(cx1_vec, cx1);
	SetCvMatFromVectors(cx2_vec, cx2);

	assert(visibleFeatureID.size() > 7);
	CvMat *F = cvCreateMat(3, 3, CV_32FC1);
	vector<int> vInlierID;
	CvMat *status = cvCreateMat(1, cx1->rows, CV_8UC1);

	// Compute F matrix
	int n = cvFindFundamentalMat(cx1, cx2, F, CV_FM_LMEDS, 1, 0.99, status);
	PrintMat(F, "Fundamental Matrix");

	// Verify correspondence based on distance between point and epipolar line
	vector<int> vCX_indx;
	CvMat *xM2 = cvCreateMat(1, 3, CV_32FC1);
	CvMat *xM1 = cvCreateMat(3, 1, CV_32FC1);
	CvMat *s = cvCreateMat(1, 1, CV_32FC1);
	for (int i = 0; i < cx1->rows; i++)
	{
		cvSetReal2D(xM2, 0, 0, cvGetReal2D(cx2, i, 0));
		cvSetReal2D(xM2, 0, 1, cvGetReal2D(cx2, i, 1));
		cvSetReal2D(xM2, 0, 2, 1);
		cvSetReal2D(xM1, 0, 0, cvGetReal2D(cx1, i, 0));
		cvSetReal2D(xM1, 1, 0, cvGetReal2D(cx1, i, 1));
		cvSetReal2D(xM1, 2, 0, 1);
		cvMatMul(xM2, F, xM2);
		cvMatMul(xM2, xM1, s);

		double l1 = cvGetReal2D(xM2, 0, 0);
		double l2 = cvGetReal2D(xM2, 0, 1);
		double l3 = cvGetReal2D(xM2, 0, 2);

		double dist = abs(cvGetReal2D(s, 0, 0)) / sqrt(l1*l1 + l2*l2);

		if (dist < 5) // 5 pixel error threshold
		{
			vInlierID.push_back(visibleFeatureID[i]);
			vCX_indx.push_back(i);
		}
	}
	cvReleaseMat(&xM2);
	cvReleaseMat(&xM1);
	cvReleaseMat(&s);

	// Remove outliers
	visibleFeatureID = vInlierID;
	CvMat *tempCx1 = cvCreateMat(vCX_indx.size(), 2, CV_32FC1);
	CvMat *tempCx2 = cvCreateMat(vCX_indx.size(), 2, CV_32FC1);
	for (int iInlier = 0; iInlier < vInlierID.size(); iInlier++)
	{
		cvSetReal2D(tempCx1, iInlier, 0, cvGetReal2D(cx1, vCX_indx[iInlier], 0));
		cvSetReal2D(tempCx1, iInlier, 1, cvGetReal2D(cx1, vCX_indx[iInlier], 1));
		cvSetReal2D(tempCx2, iInlier, 0, cvGetReal2D(cx2, vCX_indx[iInlier], 0));
		cvSetReal2D(tempCx2, iInlier, 1, cvGetReal2D(cx2, vCX_indx[iInlier], 1));
	}

	cvReleaseMat(&cx1);
	cvReleaseMat(&cx2);
	cx1 = cvCloneMat(tempCx1);
	cx2 = cvCloneMat(tempCx2);
	cvReleaseMat(&status);
	cvReleaseMat(&tempCx1);
	cvReleaseMat(&tempCx2);

	// Compute E matrix
	CvMat *E = cvCreateMat(3, 3, CV_32FC1);
	CvMat *temp33 = cvCreateMat(3, 3, CV_32FC1);
	CvMat *temp34 = cvCreateMat(3, 4, CV_32FC1);
	vector<int> ::const_iterator it1 = find(camera.vTakenFrame.begin(), camera.vTakenFrame.end(), initialFrame1);
	vector<int> ::const_iterator it2 = find(camera.vTakenFrame.begin(), camera.vTakenFrame.end(), initialFrame2);
	int idx1 = (int)(it1 - camera.vTakenFrame.begin());
	int idx2 = (int)(it2 - camera.vTakenFrame.begin());
	CvMat *K = cvCloneMat(camera.K);

	cvTranspose(K, temp33);
	cvMatMul(temp33, F, temp33);
	cvMatMul(temp33, K, E);

	// Normalize correspondences
	CvMat *invK = cvCreateMat(3, 3, CV_32FC1);
	cvInvert(K, invK);
	CvMat *nx1 = cvCreateMat(cx1->rows, cx1->cols, CV_32FC1);
	CvMat *nx2 = cvCreateMat(cx2->rows, cx2->cols, CV_32FC1);
	Pxx_inhomo(invK, cx1, nx1);
	Pxx_inhomo(invK, cx2, nx2);

	// Extract camera pose from E matrix
	GetExtrinsicParameterFromE(E, nx1, nx2, P);
	CvMat *P0 = cvCreateMat(3, 4, CV_32FC1);
	cvSetIdentity(P0);

	// Set camera projection matrices
	cvMatMul(K, P0, P0);
	cvMatMul(K, P, P);

	CvMat *cX = cvCreateMat(nx1->rows, 3, CV_32FC1);

	// Triangulation
	LinearTriangulation(cx1, P0, cx2, P, cX);
	vector<int> vI, vII;
	for (int i = 0; i < cX->rows; i++)
	{
		CvMat *X3d = cvCreateMat(4, 1, CV_32FC1);
		cvSetReal2D(X3d, 0, 0, cvGetReal2D(cX, i, 0));
		cvSetReal2D(X3d, 1, 0, cvGetReal2D(cX, i, 1));
		cvSetReal2D(X3d, 2, 0, cvGetReal2D(cX, i, 2));
		cvSetReal2D(X3d, 3, 0, 1);
		CvMat *x11 = cvCreateMat(3, 1, CV_32FC1);
		CvMat *x21 = cvCreateMat(3, 1, CV_32FC1);
		cvMatMul(P0, X3d, x11);
		cvMatMul(P, X3d, x21);
		if ((cvGetReal2D(x11, 2, 0)>0) && (cvGetReal2D(x21, 2, 0)>0))
		{
			vI.push_back(visibleFeatureID[i]);
			vII.push_back(i);
		}
	}
	CvMat *X_in = cvCreateMat(vI.size(), 3, CV_32FC1);
	for (int i = 0; i < vI.size(); i++)
	{
		cvSetReal2D(X_in, i, 0, cvGetReal2D(cX, vII[i], 0));
		cvSetReal2D(X_in, i, 1, cvGetReal2D(cX, vII[i], 1));
		cvSetReal2D(X_in, i, 2, cvGetReal2D(cX, vII[i], 2));
	}
	visibleFeatureID = vI;

	//PrintMat(cX);
	cvSetZero(X);
	SetIndexedMatRowwise(X, visibleFeatureID, X_in);
	cvReleaseMat(&X_in);


	for (int i = 0; i < visibleFeatureID.size(); i++)
	{
		vFeature[visibleFeatureID[i]].isRegistered = true;
	}

	cvReleaseMat(&F);
	cvReleaseMat(&E);
	cvReleaseMat(&temp33);
	cvReleaseMat(&temp34);
	cvReleaseMat(&K);
	cvReleaseMat(&invK);
	cvReleaseMat(&P0);
	cvReleaseMat(&cx1);
	cvReleaseMat(&cx2);
	cvReleaseMat(&nx1);
	cvReleaseMat(&nx2);
	cvReleaseMat(&cX);
	return vInlierID.size();
}

int GetUnreconstructedCorrespondences(vector<Feature> &vFeature, int frame1, int frame2, vector<int> vFeatureID, vector<vector<double> > &cx1, vector<vector<double> > &cx2, vector<int> &visibleID)
{
	visibleID.clear();

	for (int iF = 0; iF < vFeatureID.size(); iF++)
	{
		int iFeature = vFeatureID[iF];
		//cout << iFeature << " " << vFeature.size() << endl;

		if (vFeature[iFeature].isRegistered)
			continue;
		vector<int>::iterator it1 = find(vFeature[iFeature].vFrame.begin(), vFeature[iFeature].vFrame.end(), frame1);
		if (it1 == vFeature[iFeature].vFrame.end())
			continue;
		vector<int>::iterator it2 = find(vFeature[iFeature].vFrame.begin(), vFeature[iFeature].vFrame.end(), frame2);
		if (it2 == vFeature[iFeature].vFrame.end())
			continue;

		vector<double> cx1_vec, cx2_vec;

		int idx1 = int(it1 - vFeature[iFeature].vFrame.begin());
		int idx2 = int(it2 - vFeature[iFeature].vFrame.begin());

		cx1_vec.push_back(vFeature[iFeature].vx[idx1]);
		cx1_vec.push_back(vFeature[iFeature].vy[idx1]);

		cx2_vec.push_back(vFeature[iFeature].vx[idx2]);
		cx2_vec.push_back(vFeature[iFeature].vy[idx2]);

		cx1.push_back(cx1_vec);
		cx2.push_back(cx2_vec);
		visibleID.push_back(vFeature[iFeature].id);
	}

	if (visibleID.size() == 0)
	{
		return 0;
	}

	return cx1.size();
}


int CameraRegistration(CvMat *X, CvMat *x, CvMat *K, CvMat *P, double ransacThreshold, int ransacMaxIter, vector<int> &vInlierOut)
{
	int min_set = 4;
	if (X->rows < min_set)
		return 0;

	/////////////////////////////////////////////////////////////////
	// Ransac
	vector<int> vInlierIndex, vOutlierIndex;
	vInlierIndex.clear();
	vOutlierIndex.clear();

	vector<int> vInlier, vOutlier;
	int maxInlier = 0;

	CvMat *X_homoT = cvCreateMat(4, X->rows, CV_32FC1);
	CvMat *X_homo = cvCreateMat(X->rows, 4, CV_32FC1);
	CvMat *x_homoT = cvCreateMat(3, x->rows, CV_32FC1);
	CvMat *x_homo = cvCreateMat(x->rows, 3, CV_32FC1);
	Inhomo2Homo(X, X_homo);
	cvTranspose(X_homo, X_homoT);
	Inhomo2Homo(x, x_homo);
	cvTranspose(x_homo, x_homoT);

	CvMat *randx = cvCreateMat(min_set, 2, CV_32FC1);
	CvMat *randX = cvCreateMat(min_set, 3, CV_32FC1);
	CvMat *randP = cvCreateMat(3, 4, CV_32FC1);
	int *randIdx = (int *)malloc(min_set * sizeof(int));

	CvMat *reproj = cvCreateMat(3, 1, CV_32FC1);
	CvMat *homo_X = cvCreateMat(4, 1, CV_32FC1);
	for (int iRansacIter = 0; iRansacIter < ransacMaxIter; iRansacIter++)
	{
		for (int iIdx = 0; iIdx < min_set; iIdx++)
			randIdx[iIdx] = rand() % X->rows;

		for (int iIdx = 0; iIdx < min_set; iIdx++)
		{
			cvSetReal2D(randx, iIdx, 0, cvGetReal2D(x, randIdx[iIdx], 0));
			cvSetReal2D(randx, iIdx, 1, cvGetReal2D(x, randIdx[iIdx], 1));
			cvSetReal2D(randX, iIdx, 0, cvGetReal2D(X, randIdx[iIdx], 0));
			cvSetReal2D(randX, iIdx, 1, cvGetReal2D(X, randIdx[iIdx], 1));
			cvSetReal2D(randX, iIdx, 2, cvGetReal2D(X, randIdx[iIdx], 2));
		}
		EPNP_ExtrinsicCameraParamEstimation(randX, randx, K, randP);

		vInlier.clear();
		vOutlier.clear();
		for (int ip = 0; ip < X->rows; ip++)
		{
			cvSetReal2D(homo_X, 0, 0, cvGetReal2D(X, ip, 0));
			cvSetReal2D(homo_X, 1, 0, cvGetReal2D(X, ip, 1));
			cvSetReal2D(homo_X, 2, 0, cvGetReal2D(X, ip, 2));
			cvSetReal2D(homo_X, 3, 0, 1);

			cvMatMul(randP, homo_X, reproj);
			double u = cvGetReal2D(reproj, 0, 0) / cvGetReal2D(reproj, 2, 0);
			double v = cvGetReal2D(reproj, 1, 0) / cvGetReal2D(reproj, 2, 0);
			double dist = sqrt((u - cvGetReal2D(x, ip, 0))*(u - cvGetReal2D(x, ip, 0)) + (v - cvGetReal2D(x, ip, 1))*(v - cvGetReal2D(x, ip, 1)));
			if (dist < ransacThreshold)
			{
				vInlier.push_back(ip);
			}
			else
			{
				vOutlier.push_back(ip);
			}
		}

		if (vInlier.size() > maxInlier)
		{
			maxInlier = vInlier.size();
			SetSubMat(P, 0, 0, randP);
			vInlierIndex = vInlier;
			vOutlierIndex = vOutlier;
		}

		if (vInlier.size() > X->rows * 0.8)
		{
			break;
		}
	}

	vInlierOut = vInlierIndex;
	CvMat *Xin = cvCreateMat(vInlierIndex.size(), 3, CV_32FC1);
	CvMat *xin = cvCreateMat(vInlierIndex.size(), 2, CV_32FC1);
	for (int iInlier = 0; iInlier < vInlierIndex.size(); iInlier++)
	{
		cvSetReal2D(Xin, iInlier, 0, cvGetReal2D(X, vInlierIndex[iInlier], 0));
		cvSetReal2D(Xin, iInlier, 1, cvGetReal2D(X, vInlierIndex[iInlier], 1));
		cvSetReal2D(Xin, iInlier, 2, cvGetReal2D(X, vInlierIndex[iInlier], 2));

		cvSetReal2D(xin, iInlier, 0, cvGetReal2D(x, vInlierIndex[iInlier], 0));
		cvSetReal2D(xin, iInlier, 1, cvGetReal2D(x, vInlierIndex[iInlier], 1));
	}
	//EPNP_ExtrinsicCameraParamEstimation(Xin, xin, K, P);

	cvReleaseMat(&Xin);
	cvReleaseMat(&xin);
	cvReleaseMat(&reproj);
	cvReleaseMat(&homo_X);
	free(randIdx);
	cvReleaseMat(&randx);
	cvReleaseMat(&randX);
	cvReleaseMat(&randP);

	cvReleaseMat(&X_homoT);
	cvReleaseMat(&x_homo);
	cvReleaseMat(&x_homoT);
	cvReleaseMat(&X_homo);
	if (vInlierIndex.size() < 30)
		return 0;
	cout << "Number of features ePnP: " << vInlierIndex.size() << endl;
	return vInlierIndex.size();
}

int EPNP_ExtrinsicCameraParamEstimation(CvMat *X, CvMat *x, CvMat *K, CvMat *P)
{
	epnp PnP;

	PnP.set_internal_parameters(cvGetReal2D(K, 0, 2), cvGetReal2D(K, 1, 2), cvGetReal2D(K, 0, 0), cvGetReal2D(K, 1, 1));
	PnP.set_maximum_number_of_correspondences(X->rows);
	PnP.reset_correspondences();
	for (int i = 0; i < X->rows; i++) {
		PnP.add_correspondence(cvGetReal2D(X, i, 0), cvGetReal2D(X, i, 1), cvGetReal2D(X, i, 2), cvGetReal2D(x, i, 0), cvGetReal2D(x, i, 1));
	}

	double R_est[3][3], t_est[3];
	double err2 = PnP.compute_pose(R_est, t_est);

	cvSetReal2D(P, 0, 3, t_est[0]);
	cvSetReal2D(P, 1, 3, t_est[1]);
	cvSetReal2D(P, 2, 3, t_est[2]);

	cvSetReal2D(P, 0, 0, R_est[0][0]);		cvSetReal2D(P, 0, 1, R_est[0][1]);		cvSetReal2D(P, 0, 2, R_est[0][2]);
	cvSetReal2D(P, 1, 0, R_est[1][0]);		cvSetReal2D(P, 1, 1, R_est[1][1]);		cvSetReal2D(P, 1, 2, R_est[1][2]);
	cvSetReal2D(P, 2, 0, R_est[2][0]);		cvSetReal2D(P, 2, 1, R_est[2][1]);		cvSetReal2D(P, 2, 2, R_est[2][2]);
	cvMatMul(K, P, P);

	return 1;
}


int LinearTriangulation(CvMat *x1, CvMat *P1, CvMat *x2, CvMat *P2, vector<int> &featureID, vector<vector<double> > &X, vector<int> &filteredFeatureID)
{
	filteredFeatureID.clear();
	CvMat *A = cvCreateMat(4, 4, CV_32FC1);
	CvMat *A1 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *A2 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *A3 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *A4 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *P1_1 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *P1_2 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *P1_3 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *P2_1 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *P2_2 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *P2_3 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *temp14_1 = cvCreateMat(1, 4, CV_32FC1);
	CvMat *x = cvCreateMat(A->cols, 1, CV_32FC1);

	for (int ix = 0; ix < x1->rows; ix++)
	{
		GetSubMatRowwise(P1, 0, 0, P1_1);
		GetSubMatRowwise(P1, 1, 1, P1_2);
		GetSubMatRowwise(P1, 2, 2, P1_3);
		GetSubMatRowwise(P2, 0, 0, P2_1);
		GetSubMatRowwise(P2, 1, 1, P2_2);
		GetSubMatRowwise(P2, 2, 2, P2_3);

		ScalarMul(P1_3, cvGetReal2D(x1, ix, 0), temp14_1);
		cvSub(temp14_1, P1_1, A1);

		ScalarMul(P1_3, cvGetReal2D(x1, ix, 1), temp14_1);
		cvSub(temp14_1, P1_2, A2);

		ScalarMul(P2_3, cvGetReal2D(x2, ix, 0), temp14_1);
		cvSub(temp14_1, P2_1, A3);
		ScalarMul(P2_3, cvGetReal2D(x2, ix, 1), temp14_1);
		cvSub(temp14_1, P2_2, A4);
		SetSubMat(A, 0, 0, A1);
		SetSubMat(A, 1, 0, A2);
		SetSubMat(A, 2, 0, A3);
		SetSubMat(A, 3, 0, A4);

		LS_homogeneous(A, x);

		double v = cvGetReal2D(x, 3, 0);
		if (abs(v) < POINT_AT_INFINITY_ZERO)
		{
			continue;
		}

		vector<double> X_vec;
		X_vec.push_back(cvGetReal2D(x, 0, 0) / v);
		X_vec.push_back(cvGetReal2D(x, 1, 0) / v);
		X_vec.push_back(cvGetReal2D(x, 2, 0) / v);
		X.push_back(X_vec);
		filteredFeatureID.push_back(featureID[ix]);
	}

	cvReleaseMat(&A);
	cvReleaseMat(&A1);
	cvReleaseMat(&A2);
	cvReleaseMat(&A3);
	cvReleaseMat(&A4);
	cvReleaseMat(&P1_1);
	cvReleaseMat(&P1_2);
	cvReleaseMat(&P1_3);
	cvReleaseMat(&P2_1);
	cvReleaseMat(&P2_2);
	cvReleaseMat(&P2_3);
	cvReleaseMat(&temp14_1);
	cvReleaseMat(&x);

	if (filteredFeatureID.size() == 0)
		return 0;

	return X.size();
}


int ExcludePointBehindCamera(CvMat *X, CvMat *P1, CvMat *P2, vector<int> &featureID, vector<int> &excludedFeatureID, vector<vector<double> > &cX)
{
	excludedFeatureID.clear();
	for (int i = 0; i < X->rows; i++)
	{
		CvMat *X_3d = cvCreateMat(4, 1, CV_32FC1);
		cvSetReal2D(X_3d, 0, 0, cvGetReal2D(X, i, 0));
		cvSetReal2D(X_3d, 1, 0, cvGetReal2D(X, i, 1));
		cvSetReal2D(X_3d, 2, 0, cvGetReal2D(X, i, 2));
		cvSetReal2D(X_3d, 3, 0, 1);

		CvMat *x1 = cvCreateMat(3, 1, CV_32FC1);
		CvMat *x2 = cvCreateMat(3, 1, CV_32FC1);

		cvMatMul(P1, X_3d, x1);
		cvMatMul(P2, X_3d, x2);

		if ((cvGetReal2D(x1, 2, 0) > 0) && (cvGetReal2D(x2, 2, 0) > 0))
		{
			excludedFeatureID.push_back(featureID[i]);

			vector<double> cX_vec;
			cX_vec.push_back(cvGetReal2D(X, i, 0));
			cX_vec.push_back(cvGetReal2D(X, i, 1));
			cX_vec.push_back(cvGetReal2D(X, i, 2));

			cX.push_back(cX_vec);
		}

		cvReleaseMat(&x1);
		cvReleaseMat(&x2);
		cvReleaseMat(&X_3d);
	}
	if (excludedFeatureID.size() == 0)
	{
		return 0;
	}
	return cX.size();
}

int ExcludePointAtInfinity(CvMat *X, CvMat *P1, CvMat *P2, CvMat *K, vector<int> &featureID, vector<int> &excludedFeatureID, vector<vector<double> > &cX)
{
	CvMat *q1 = cvCreateMat(4, 1, CV_32FC1);
	CvMat *R1 = cvCreateMat(3, 3, CV_32FC1);
	CvMat *t1 = cvCreateMat(3, 1, CV_32FC1);
	CvMat *invK = cvCreateMat(3, 3, CV_32FC1);
	CvMat *invR1 = cvCreateMat(3, 3, CV_32FC1);

	CvMat *q2 = cvCreateMat(4, 1, CV_32FC1);
	CvMat *R2 = cvCreateMat(3, 3, CV_32FC1);
	CvMat *t2 = cvCreateMat(3, 1, CV_32FC1);
	CvMat *invR2 = cvCreateMat(3, 3, CV_32FC1);

	GetSubMatColwise(P1, 0, 2, R1);
	GetSubMatColwise(P1, 3, 3, t1);
	cvInvert(K, invK);
	cvMatMul(invK, R1, R1);
	cvInvert(R1, invR1);
	cvMatMul(invK, t1, t1);
	cvMatMul(invR1, t1, t1);
	ScalarMul(t1, -1, t1);

	GetSubMatColwise(P2, 0, 2, R2);
	GetSubMatColwise(P2, 3, 3, t2);
	cvMatMul(invK, R2, R2);
	cvInvert(R2, invR2);
	cvMatMul(invK, t2, t2);
	cvMatMul(invR2, t2, t2);
	ScalarMul(t2, -1, t2);

	double xC1 = cvGetReal2D(t1, 0, 0);
	double yC1 = cvGetReal2D(t1, 1, 0);
	double zC1 = cvGetReal2D(t1, 2, 0);
	double xC2 = cvGetReal2D(t2, 0, 0);
	double yC2 = cvGetReal2D(t2, 1, 0);
	double zC2 = cvGetReal2D(t2, 2, 0);

	excludedFeatureID.clear();
	vector<double> vInner;
	for (int i = 0; i < X->rows; i++)
	{
		double x3D = cvGetReal2D(X, i, 0);
		double y3D = cvGetReal2D(X, i, 1);
		double z3D = cvGetReal2D(X, i, 2);

		double v1x = x3D - xC1;		double v1y = y3D - yC1;		double v1z = z3D - zC1;
		double v2x = x3D - xC2;		double v2y = y3D - yC2;		double v2z = z3D - zC2;

		double nv1 = sqrt(v1x*v1x + v1y*v1y + v1z*v1z);
		double nv2 = sqrt(v2x*v2x + v2y*v2y + v2z*v2z);
		v1x /= nv1;		v1y /= nv1;		v1z /= nv1;
		v2x /= nv2;		v2y /= nv2;		v2z /= nv2;
		double inner = v1x*v2x + v1y*v2y + v1z*v2z;
		vInner.push_back(inner);
		if ((abs(inner) < cos(PI / 180 * 2.0)) && (inner > 0))
		{
			vector<double> cX_vec;
			cX_vec.push_back(x3D);
			cX_vec.push_back(y3D);
			cX_vec.push_back(z3D);
			cX.push_back(cX_vec);
			excludedFeatureID.push_back(featureID[i]);
		}

	}
	cvReleaseMat(&q1);
	cvReleaseMat(&R1);
	cvReleaseMat(&t1);
	cvReleaseMat(&invK);
	cvReleaseMat(&invR1);

	cvReleaseMat(&q2);
	cvReleaseMat(&R2);
	cvReleaseMat(&t2);
	cvReleaseMat(&invR2);

	if (excludedFeatureID.size() == 0)
	{
		return 0;
	}
	
	return cX.size();
}


bool ExcludePointHighReprojectionError(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> &vUsedFrame
	, vector<int> &visibleStrucrtureID, CvMat *X_tot
	, vector<int> &visibleStrucrtureID_new, vector<vector<double> > &X_tot_new,
	double omega, CvMat *K)
{
	visibleStrucrtureID_new.clear();
	CvMat *X = cvCreateMat(4, 1, CV_32FC1);
	CvMat *x = cvCreateMat(3, 1, CV_32FC1);
	for (int iVS = 0; iVS < visibleStrucrtureID.size(); iVS++)
	{
		bool isIn = true;
		for (int iP = 0; iP < cP.size(); iP++)
		{
			vector<int>::const_iterator it = find(vFeature[visibleStrucrtureID[iVS]].vFrame.begin(), vFeature[visibleStrucrtureID[iVS]].vFrame.end(), vUsedFrame[iP]);
			if (it != vFeature[visibleStrucrtureID[iVS]].vFrame.end())
			{
				int idx = (int)(it - vFeature[visibleStrucrtureID[iVS]].vFrame.begin());

				cvSetReal2D(X, 0, 0, cvGetReal2D(X_tot, iVS, 0));
				cvSetReal2D(X, 1, 0, cvGetReal2D(X_tot, iVS, 1));
				cvSetReal2D(X, 2, 0, cvGetReal2D(X_tot, iVS, 2));

				cvSetReal2D(X, 3, 0, 1);
				cvMatMul(cP[iP], X, x);

				double u = cvGetReal2D(x, 0, 0) / cvGetReal2D(x, 2, 0);
				double v = cvGetReal2D(x, 1, 0) / cvGetReal2D(x, 2, 0);

				double tan_omega_half_2 = tan(omega / 2) * 2;

				double K11 = cvGetReal2D(K, 0, 0);
				double K22 = cvGetReal2D(K, 1, 1);
				double K13 = cvGetReal2D(K, 0, 2);
				double K23 = cvGetReal2D(K, 1, 2);

				double u_n = u / K11 - K13 / K11;
				double v_n = v / K22 - K23 / K22;

				double r_u = sqrt(u_n*u_n + v_n*v_n);
				double r_d = 1 / omega*atan(r_u*tan_omega_half_2);

				double u_d_n = r_d / r_u * u_n;
				double v_d_n = r_d / r_u * v_n;

				double u1 = u_d_n *K11 + K13;
				double v1 = v_d_n *K22 + K23;

				double u0 = vFeature[visibleStrucrtureID[iVS]].vx_dis[idx];
				double v0 = vFeature[visibleStrucrtureID[iVS]].vy_dis[idx];

				if (sqrt((u0 - u1)*(u0 - u1) + (v0 - v1)*(v0 - v1)) > 5)
				{
					isIn = false;
					break;
				}
			}
		}
		if (isIn)
		{
			visibleStrucrtureID_new.push_back(visibleStrucrtureID[iVS]);
			vector<double> X_tot_new_vec;
			X_tot_new_vec.push_back(cvGetReal2D(X_tot, iVS, 0));
			X_tot_new_vec.push_back(cvGetReal2D(X_tot, iVS, 1));
			X_tot_new_vec.push_back(cvGetReal2D(X_tot, iVS, 2));
			X_tot_new.push_back(X_tot_new_vec);
		}
	}
	cvReleaseMat(&X);
	cvReleaseMat(&x);

	if (visibleStrucrtureID_new.size() == 0)
		return false;

	return true;
}

int GetFeatureGivenFrame(vector<Feature> &vFeature, int frame1)
{
	int count = 0;
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		if (!vFeature[iFeature].isRegistered)
			continue;
		vector<int>::iterator it1 = find(vFeature[iFeature].vFrame.begin(), vFeature[iFeature].vFrame.end(), frame1);
		if (it1 == vFeature[iFeature].vFrame.end())
			continue;
		count++;
	}
	return count;
}



int ExcludePointHighReprojectionError(vector<Feature> &vFeature, vector<CvMat *> &cP, vector<int> vUsedFrame, CvMat *X_tot, double omega, CvMat *K)
{
	CvMat *X = cvCreateMat(4, 1, CV_32FC1);
	CvMat *x = cvCreateMat(3, 1, CV_32FC1);
	int count1 = 0, count2 = 0;

	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		if (vFeature[iFeature].isRegistered)
		{
			count1++;
			int nProj = 0;
			for (int iP = 0; iP < cP.size(); iP++)
			{
				vector<int>::const_iterator it = find(vFeature[iFeature].vFrame.begin(), vFeature[iFeature].vFrame.end(), vUsedFrame[iP]);
				if (it != vFeature[iFeature].vFrame.end())
				{
					nProj++;
				}
			}
			vFeature[iFeature].nProj = nProj;

			if (nProj == 0)
				continue;

			for (int iP = 0; iP < cP.size(); iP++)
			{
				vector<int>::const_iterator it = find(vFeature[iFeature].vFrame.begin(), vFeature[iFeature].vFrame.end(), vUsedFrame[iP]);
				if (it != vFeature[iFeature].vFrame.end())
				{
					int idx = (int)(it - vFeature[iFeature].vFrame.begin());

					cvSetReal2D(X, 0, 0, cvGetReal2D(X_tot, iFeature, 0));
					cvSetReal2D(X, 1, 0, cvGetReal2D(X_tot, iFeature, 1));
					cvSetReal2D(X, 2, 0, cvGetReal2D(X_tot, iFeature, 2));

					cvSetReal2D(X, 3, 0, 1);
					cvMatMul(cP[iP], X, x);

					double u = cvGetReal2D(x, 0, 0) / cvGetReal2D(x, 2, 0);
					double v = cvGetReal2D(x, 1, 0) / cvGetReal2D(x, 2, 0);

					double tan_omega_half_2 = tan(omega / 2) * 2;

					double K11 = cvGetReal2D(K, 0, 0);
					double K22 = cvGetReal2D(K, 1, 1);
					double K13 = cvGetReal2D(K, 0, 2);
					double K23 = cvGetReal2D(K, 1, 2);

					double u_n = u / K11 - K13 / K11;
					double v_n = v / K22 - K23 / K22;

					double r_u = sqrt(u_n*u_n + v_n*v_n);
					double r_d = 1 / omega*atan(r_u*tan_omega_half_2);

					double u_d_n = r_d / r_u * u_n;
					double v_d_n = r_d / r_u * v_n;

					double u1 = u_d_n*K11 + K13;
					double v1 = v_d_n*K22 + K23;

					double u0 = vFeature[iFeature].vx_dis[idx];
					double v0 = vFeature[iFeature].vy_dis[idx];


					double dist = sqrt((u0 - u1)*(u0 - u1) + (v0 - v1)*(v0 - v1));


					if (dist > 5)
					{
						vFeature[iFeature].vFrame.erase(vFeature[iFeature].vFrame.begin() + idx);
						vFeature[iFeature].vx.erase(vFeature[iFeature].vx.begin() + idx);
						vFeature[iFeature].vy.erase(vFeature[iFeature].vy.begin() + idx);
						vFeature[iFeature].vx_dis.erase(vFeature[iFeature].vx_dis.begin() + idx);
						vFeature[iFeature].vy_dis.erase(vFeature[iFeature].vy_dis.begin() + idx);
						//vFeature[iFeature].vCamera.erase(vFeature[iFeature].vCamera.begin() + idx);
						nProj--;
						if (nProj < 2)
						{
							vFeature[iFeature].isRegistered = false;
							count2++;
							break;
						}
					}

				}
			}

			vFeature[iFeature].nProj = nProj;
		}
	}
	cout << count2 << " points are deleted." << endl;
	cvReleaseMat(&X);
	cvReleaseMat(&x);

	return count1;

}

void LinearTriangulation(CvMat *x1, CvMat *P1, CvMat *x2, CvMat *P2, CvMat *X)
{
	for (int ix = 0; ix < x1->rows; ix++)
	{
		CvMat *A = cvCreateMat(4, 4, CV_32FC1);
		CvMat *A1 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *A2 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *A3 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *A4 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P1_1 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P1_2 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P1_3 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P2_1 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P2_2 = cvCreateMat(1, 4, CV_32FC1);
		CvMat *P2_3 = cvCreateMat(1, 4, CV_32FC1);

		CvMat *temp14_1 = cvCreateMat(1, 4, CV_32FC1);

		GetSubMatRowwise(P1, 0, 0, P1_1);
		GetSubMatRowwise(P1, 1, 1, P1_2);
		GetSubMatRowwise(P1, 2, 2, P1_3);
		GetSubMatRowwise(P2, 0, 0, P2_1);
		GetSubMatRowwise(P2, 1, 1, P2_2);
		GetSubMatRowwise(P2, 2, 2, P2_3);

		ScalarMul(P1_3, cvGetReal2D(x1, ix, 0), temp14_1);
		cvSub(temp14_1, P1_1, A1);

		ScalarMul(P1_3, cvGetReal2D(x1, ix, 1), temp14_1);
		cvSub(temp14_1, P1_2, A2);

		ScalarMul(P2_3, cvGetReal2D(x2, ix, 0), temp14_1);
		cvSub(temp14_1, P2_1, A3);
		ScalarMul(P2_3, cvGetReal2D(x2, ix, 1), temp14_1);
		cvSub(temp14_1, P2_2, A4);
		SetSubMat(A, 0, 0, A1);
		SetSubMat(A, 1, 0, A2);
		SetSubMat(A, 2, 0, A3);
		SetSubMat(A, 3, 0, A4);

		CvMat *x = cvCreateMat(A->cols, 1, CV_32FC1);
		LS_homogeneous(A, x);
		double v = cvGetReal2D(x, 3, 0);
		cvSetReal2D(X, ix, 0, cvGetReal2D(x, 0, 0) / v);
		cvSetReal2D(X, ix, 1, cvGetReal2D(x, 1, 0) / v);
		cvSetReal2D(X, ix, 2, cvGetReal2D(x, 2, 0) / v);

		cvReleaseMat(&A);
		cvReleaseMat(&A1);
		cvReleaseMat(&A2);
		cvReleaseMat(&A3);
		cvReleaseMat(&A4);
		cvReleaseMat(&P1_1);
		cvReleaseMat(&P1_2);
		cvReleaseMat(&P1_3);
		cvReleaseMat(&P2_1);
		cvReleaseMat(&P2_2);
		cvReleaseMat(&P2_3);
		cvReleaseMat(&temp14_1);
		cvReleaseMat(&x);
	}
}


void SetCvMatFromVectors(vector<vector<double> > x, CvMat *X)
{
	for (int i = 0; i < x.size(); i++)
	{
		for (int j = 0; j < x[i].size(); j++)
			cvSetReal2D(X, i, j, x[i][j]);
	}
}

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

#define ZERO_DISTANCE 1e+0
#define DESCRIPTOR_ZERO_DISTANCE 2e+2
#define FILE_PATH ""

void Stitching(vector<Feature> &vStitchedFeature, Feature feature, int &feature_idx, Feature &newFeature,
	int max_nCameras, int max_nFrames);
void Stitching(vector<Feature> &vFeature, vector<Feature> &vStitchedFeature,
	int max_nCameras, int max_nFrames, int iThread);
void Aggregation(string path, string input, int nFiles);
using namespace std;

bool IsSamePoint(Point p1, Point p2)
{
	if (DistancePixel(p1.x, p1.y, p2.x, p2.y) < ZERO_DISTANCE)
		return true;
	else
		return false;
}

bool IsSamePoint(vector<double> &desc1, vector<double> &desc2)
{
	double norm = 0;
	for (int i = 0; i < 128; i++)
	{
		norm = norm + (desc1[i]-desc2[i])*(desc1[i]-desc2[i]);
	}
	norm = sqrt(norm);

	if (norm < DESCRIPTOR_ZERO_DISTANCE)
		return true;
	else 
		return false;
}

void ComputeMeanDescriptor(vector<double> vMean1, vector<double> vMean2, vector<double> &vMean)
{
	for (int i = 0; i < 128; i++)
	{
		int mean1 = (int)floor((double)(vMean1[i] * vMean1[128] + vMean2[i]) / (double)(vMean1[128] + 1));
		vMean.push_back(mean1);
	}
	vMean.push_back(vMean1[128] + 1);
}

int main ( int argc, char * argv[] )
{
	////////////////////////////////////////////////////////////
	// Filename setting
	string path = FILE_PATH;
	string savepath = path + "reconstruction/";
	string savepath_m = savepath + "measurement/";
	string savefile_static = savepath_m + "static_measurement_desc%07d.txt";

	string filelist = path + "image/filelist.list";
	vector<string> vFilename;
	LoadFileListData(filelist, vFilename);

	Aggregation(savepath, savefile_static, vFilename.size());
	return 0;
} // end main()

void Aggregation(string path, string input, int nFiles)
{
	vector<Feature> vFeature, vTempFeature;
	int nFeatures, nTempFeatures;

	vector<int> vnFrames;
	string savefile = path + "stitchedmeasurement_static.txt";
	string savefile_desc = path + "descriptors.txt";

	int max_nCameras = 1;
	int nCameras = 1;

	for (int i = 0; i < nFiles; i++)
	{
		char temp[1000];
		sprintf(temp, input.c_str(), i);
		string filename = temp;
		LoadMeasurementData_RGB_DESC_Seq(filename, vFeature);
	}
	int max_nFrames = 0;
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		for (int iFrame = 0; iFrame < vFeature[iFeature].vFrame.size(); iFrame++)
		{
			if (max_nFrames < vFeature[iFeature].vFrame[iFrame])
				max_nFrames = vFeature[iFeature].vFrame[iFrame];
		}
	}
	max_nFrames++;
	nFeatures = vFeature.size();
	vnFrames.push_back(max_nFrames);

	vector<vector<Feature> > vvFeature;

	int nThread = omp_get_max_threads()-1;
	int nFeature_each = vFeature.size()/nThread;

	int iFeature_num = 0;
	for (int i = 0; i < nThread-1; i++)
	{
		vector<Feature> vFeature1;
		for (int iFeature = 0; iFeature < nFeature_each; iFeature++)
		{
			vFeature1.push_back(vFeature[iFeature_num]);
			iFeature_num++;
		}
		vvFeature.push_back(vFeature1);
	}

	vector<Feature> vFeature1;
	vFeature1.clear();
	for (int i = iFeature_num; i < vFeature.size(); i++)
	{
		vFeature1.push_back(vFeature[i]);
	}
	vvFeature.push_back(vFeature1);
	vFeature.clear();
	vFeature1.clear();

	vector<vector<Feature> > vvStitchedFeature;
	vvStitchedFeature.resize(vvFeature.size());
	int nTotal_feature = 0;
	#pragma omp parallel for 
	for (int iThread = 0; iThread < vvFeature.size(); iThread++)
	{
		vector<Feature> vStitchedFeature1;
		Stitching(vvFeature[iThread], vStitchedFeature1, max_nCameras, max_nFrames,iThread);
		vvStitchedFeature[iThread] = vStitchedFeature1;
	}

	for (int iThread = 0; iThread < nThread; iThread++)
	{
		nTotal_feature += vvStitchedFeature[iThread].size();
	}
	nTotal_feature -= vvStitchedFeature[0].size();

	vector<Feature> vStitchedFeature, vFeature_new;
	vStitchedFeature = vvStitchedFeature[0];
	int count = 0;
	for (int i = 1; i < vvStitchedFeature.size(); i++)
	{
		vector<int> vIdx;
		vector<Feature> vFeature_1;
		vIdx.resize(vvStitchedFeature[i].size());
		vFeature_1.resize(vvStitchedFeature[i].size());

		#pragma omp parallel for 
		for (int iFeature = 0; iFeature < vvStitchedFeature[i].size(); iFeature++)
		{
			//int th_id = omp_get_thread_num();
			int idx = -1;
			Feature fs; 
			Stitching(vStitchedFeature, vvStitchedFeature[i][iFeature], idx, fs, max_nCameras, max_nFrames);
		
			vIdx[iFeature] = idx;
			vFeature_1[iFeature] = fs;
			count++;
			if (count%1000 ==0)
				cout << count << " " << nTotal_feature << endl;;
		}

		for (int iIdx = 0; iIdx < vIdx.size(); iIdx++)
		{
			//if (iIdx < 100)
			//	cout << vIdx[iIdx] << " ";
			if (vIdx[iIdx] != -1)
				vStitchedFeature[vIdx[iIdx]] = vFeature_1[iIdx];
			else
				vStitchedFeature.push_back(vFeature_1[iIdx]);
		}
		//break;
	}


	SaveMeasurementData_RGB(savefile, vStitchedFeature, max_nFrames, FILESAVE_WRITE_MODE);
	SaveMeasurementData_DESC(savefile_desc, vStitchedFeature, max_nFrames, FILESAVE_WRITE_MODE);
	return;
}

void Stitching(vector<Feature> &vFeature, vector<Feature> &vStitchedFeature,
				int max_nCameras, int max_nFrames, int iThread)
{
	for (int iFeature = 0; iFeature < vFeature.size(); iFeature++)
	{
		if (iFeature % 1000 == 1)
			cout << iThread << " / " << iFeature << " / " << vFeature.size() << endl;
		bool isInSet = false;
		for (int iStitchedFeature = 0; iStitchedFeature < vStitchedFeature.size(); iStitchedFeature++)
		{
			if (isInSet)
				break;
			if (vStitchedFeature[iStitchedFeature].vFrame[0] > vFeature[iFeature].vFrame[vFeature[iFeature].vFrame.size()-1])
				continue;
			if (vStitchedFeature[iStitchedFeature].vFrame[vStitchedFeature[iStitchedFeature].vFrame.size()-1] < vFeature[iFeature].vFrame[0])
				continue;

			// Find the same frame
			for (int iFeatureFrame = 0; iFeatureFrame < vFeature[iFeature].vFrame.size(); iFeatureFrame++)
			{
				if ((vFeature[iFeature].vFrame[iFeatureFrame] < vStitchedFeature[iStitchedFeature].vFrame[0])||
					(vFeature[iFeature].vFrame[iFeatureFrame] > vStitchedFeature[iStitchedFeature].vFrame[vStitchedFeature[iStitchedFeature].vFrame.size()-1]))
					continue;

				vector<int>::const_iterator it = find(vStitchedFeature[iStitchedFeature].vFrame.begin(),vStitchedFeature[iStitchedFeature].vFrame.end(), 
					vFeature[iFeature].vFrame[iFeatureFrame]);
				if (it != vStitchedFeature[iStitchedFeature].vFrame.end())
				{
					int idx = (int) (it - vStitchedFeature[iStitchedFeature].vFrame.begin());
					Point pFeature, pStitchedFeature;
					pFeature.x = vFeature[iFeature].vx[iFeatureFrame];
					pFeature.y = vFeature[iFeature].vy[iFeatureFrame];
					pStitchedFeature.x = vStitchedFeature[iStitchedFeature].vx[idx];
					pStitchedFeature.y = vStitchedFeature[iStitchedFeature].vy[idx];
					//if (IsSamePoint(pFeature, pStitchedFeature, vFeature[iFeature].vvDesc[0], vStitchedFeature[iStitchedFeature].vvDesc[0]))
					if (IsSamePoint(pFeature, pStitchedFeature))
					{
						//if (IsSamePoint(vFeature[iFeature].vvDesc[0], vStitchedFeature[iStitchedFeature].vvDesc[0]))
						{
							Feature fs;
							fs.id = vStitchedFeature[iStitchedFeature].id;
							fs.r = vStitchedFeature[iStitchedFeature].r;
							fs.g = vStitchedFeature[iStitchedFeature].g;
							fs.b = vStitchedFeature[iStitchedFeature].b;
							int temp = 0;
							for (int iTotalFrame = 0; iTotalFrame < max_nCameras*max_nFrames; iTotalFrame++)
							{
								vector<int>::const_iterator it_total1 = find(vFeature[iFeature].vFrame.begin(),vFeature[iFeature].vFrame.end(), iTotalFrame);
								vector<int>::const_iterator it_total2 = find(vStitchedFeature[iStitchedFeature].vFrame.begin(),vStitchedFeature[iStitchedFeature].vFrame.end(), iTotalFrame);
								if ((it_total1 != vFeature[iFeature].vFrame.end()) && (it_total2 != vStitchedFeature[iStitchedFeature].vFrame.end()))
								{
									int idx_total = (int) (it_total2 - vStitchedFeature[iStitchedFeature].vFrame.begin());
									fs.vFrame.push_back(vStitchedFeature[iStitchedFeature].vFrame[idx_total]);
									fs.vCamera.push_back(vStitchedFeature[iStitchedFeature].vCamera[idx_total]);
									fs.vx.push_back(vStitchedFeature[iStitchedFeature].vx[idx_total]);
									fs.vy.push_back(vStitchedFeature[iStitchedFeature].vy[idx_total]);
									fs.vx_dis.push_back(vStitchedFeature[iStitchedFeature].vx_dis[idx_total]);
									fs.vy_dis.push_back(vStitchedFeature[iStitchedFeature].vy_dis[idx_total]);
									vector<double> meanDesc;
									if ((fs.vvDesc.size() == 0) || (vStitchedFeature[iStitchedFeature].vvDesc.size() == 0))
									{
										if (fs.vvDesc.size() == 0)
										{
											meanDesc = vStitchedFeature[iStitchedFeature].vvDesc[0];
										}
										else
										{
											meanDesc = fs.vvDesc[0];
										}
									}
									else
									{
										ComputeMeanDescriptor(fs.vvDesc[0], vStitchedFeature[iStitchedFeature].vvDesc[0], meanDesc);
									}							
									fs.vvDesc.clear();
									fs.vvDesc.push_back(meanDesc);
								}
								else if ((it_total1 != vFeature[iFeature].vFrame.end()) && (it_total2 == vStitchedFeature[iStitchedFeature].vFrame.end()))
								{
									int idx_total = (int) (it_total1 - vFeature[iFeature].vFrame.begin());
									fs.vFrame.push_back(vFeature[iFeature].vFrame[idx_total]);
									fs.vCamera.push_back(vFeature[iFeature].vCamera[idx_total]);
									fs.vx.push_back(vFeature[iFeature].vx[idx_total]);
									fs.vy.push_back(vFeature[iFeature].vy[idx_total]);
									fs.vx_dis.push_back(vFeature[iFeature].vx_dis[idx_total]);
									fs.vy_dis.push_back(vFeature[iFeature].vy_dis[idx_total]);
									//fs.vvDesc.push_back(vFeature[iFeature].vvDesc[idx_total]);
									vector<double> meanDesc;
									if ((fs.vvDesc.size() == 0) || (vFeature[iFeature].vvDesc.size() == 0))
									{
										if (fs.vvDesc.size() == 0)
										{
											meanDesc = vFeature[iFeature].vvDesc[0];
										}
										else
										{
											meanDesc = fs.vvDesc[0];
										}
									}
									else
									{
										ComputeMeanDescriptor(fs.vvDesc[0], vFeature[iFeature].vvDesc[0], meanDesc);
									}							
									fs.vvDesc.clear();
									fs.vvDesc.push_back(meanDesc);
								}
								else if ((it_total1 == vFeature[iFeature].vFrame.end()) && (it_total2 != vStitchedFeature[iStitchedFeature].vFrame.end()))
								{
									int idx_total = (int) (it_total2 - vStitchedFeature[iStitchedFeature].vFrame.begin());
									fs.vFrame.push_back(vStitchedFeature[iStitchedFeature].vFrame[idx_total]);
									fs.vCamera.push_back(vStitchedFeature[iStitchedFeature].vCamera[idx_total]);
									fs.vx.push_back(vStitchedFeature[iStitchedFeature].vx[idx_total]);
									fs.vy.push_back(vStitchedFeature[iStitchedFeature].vy[idx_total]);
									fs.vx_dis.push_back(vStitchedFeature[iStitchedFeature].vx_dis[idx_total]);
									fs.vy_dis.push_back(vStitchedFeature[iStitchedFeature].vy_dis[idx_total]);
									//fs.vvDesc.push_back(vStitchedFeature[iStitchedFeature].vvDesc[idx_total]);
									vector<double> meanDesc;
									if ((fs.vvDesc.size() == 0) || (vStitchedFeature[iStitchedFeature].vvDesc.size() == 0))
									{
										if (fs.vvDesc.size() == 0)
										{
											meanDesc = vStitchedFeature[iStitchedFeature].vvDesc[0];
										}
										else
										{
											meanDesc = fs.vvDesc[0];
										}
									}
									else
									{
										ComputeMeanDescriptor(fs.vvDesc[0], vStitchedFeature[iStitchedFeature].vvDesc[0], meanDesc);
									}							
									fs.vvDesc.clear();
									fs.vvDesc.push_back(meanDesc);
								}
								temp = iTotalFrame;
							}

							//if (fs.vFrame.size() == 0)
							//{
							//	cout << endl << temp << endl;
							//	for (int i =0; i < vStitchedFeature[iStitchedFeature].vFrame.size(); i++)
							//		cout << vStitchedFeature[iStitchedFeature].vFrame[i] << " ";
							//}
							vStitchedFeature[iStitchedFeature] = fs;
							isInSet = true;
						}
					}
					if (isInSet)
						break;
				}
			}
		}
		if (!isInSet)
			vStitchedFeature.push_back(vFeature[iFeature]);
	}
}

void Stitching(vector<Feature> &vStitchedFeature, Feature feature, int &feature_idx, Feature &newFeature,
				int max_nCameras, int max_nFrames)
{
	bool isInSet = false;
	for (int iStitchedFeature = 0; iStitchedFeature < vStitchedFeature.size(); iStitchedFeature++)
	{
		if (isInSet)
			break;

		// Find the same frame
		for (int iFeatureFrame = 0; iFeatureFrame < feature.vFrame.size(); iFeatureFrame++)
		{
			vector<int>::const_iterator it = find(vStitchedFeature[iStitchedFeature].vFrame.begin(),vStitchedFeature[iStitchedFeature].vFrame.end(), 
				feature.vFrame[iFeatureFrame]);
			if (it != vStitchedFeature[iStitchedFeature].vFrame.end())
			{
				int idx = (int) (it - vStitchedFeature[iStitchedFeature].vFrame.begin());
				Point pFeature, pStitchedFeature;
				pFeature.x = feature.vx[iFeatureFrame];
				pFeature.y = feature.vy[iFeatureFrame];
				pStitchedFeature.x = vStitchedFeature[iStitchedFeature].vx[idx];
				pStitchedFeature.y = vStitchedFeature[iStitchedFeature].vy[idx];
				//if (IsSamePoint(pFeature, pStitchedFeature, vFeature[iFeature].vvDesc[0], vStitchedFeature[iStitchedFeature].vvDesc[0]))
				if (IsSamePoint(pFeature, pStitchedFeature))
				{
					//if (IsSamePoint(vFeature[iFeature].vvDesc[0], vStitchedFeature[iStitchedFeature].vvDesc[0]))
					{
						Feature fs;
						fs.id = vStitchedFeature[iStitchedFeature].id;
						fs.r = vStitchedFeature[iStitchedFeature].r;
						fs.g = vStitchedFeature[iStitchedFeature].g;
						fs.b = vStitchedFeature[iStitchedFeature].b;
						int temp = 0;
						for (int iTotalFrame = 0; iTotalFrame < max_nCameras*max_nFrames; iTotalFrame++)
						{
							vector<int>::const_iterator it_total1 = find(feature.vFrame.begin(),feature.vFrame.end(), iTotalFrame);
							vector<int>::const_iterator it_total2 = find(vStitchedFeature[iStitchedFeature].vFrame.begin(),vStitchedFeature[iStitchedFeature].vFrame.end(), iTotalFrame);
							if ((it_total1 != feature.vFrame.end()) && (it_total2 != vStitchedFeature[iStitchedFeature].vFrame.end()))
							{
								int idx_total = (int) (it_total2 - vStitchedFeature[iStitchedFeature].vFrame.begin());
								fs.vFrame.push_back(vStitchedFeature[iStitchedFeature].vFrame[idx_total]);
								fs.vCamera.push_back(vStitchedFeature[iStitchedFeature].vCamera[idx_total]);
								fs.vx.push_back(vStitchedFeature[iStitchedFeature].vx[idx_total]);
								fs.vy.push_back(vStitchedFeature[iStitchedFeature].vy[idx_total]);
								fs.vx_dis.push_back(vStitchedFeature[iStitchedFeature].vx_dis[idx_total]);
								fs.vy_dis.push_back(vStitchedFeature[iStitchedFeature].vy_dis[idx_total]);
								vector<double> meanDesc;
								if ((fs.vvDesc.size() == 0) || (vStitchedFeature[iStitchedFeature].vvDesc.size() == 0))
								{
									if (fs.vvDesc.size() == 0)
									{
										meanDesc = vStitchedFeature[iStitchedFeature].vvDesc[0];
									}
									else
									{
										meanDesc = fs.vvDesc[0];
									}
								}
								else
								{
									ComputeMeanDescriptor(fs.vvDesc[0], vStitchedFeature[iStitchedFeature].vvDesc[0], meanDesc);
								}							
								fs.vvDesc.clear();
								fs.vvDesc.push_back(meanDesc);
							}
							else if ((it_total1 != feature.vFrame.end()) && (it_total2 == vStitchedFeature[iStitchedFeature].vFrame.end()))
							{
								int idx_total = (int) (it_total1 - feature.vFrame.begin());
								fs.vFrame.push_back(feature.vFrame[idx_total]);
								fs.vCamera.push_back(feature.vCamera[idx_total]);
								fs.vx.push_back(feature.vx[idx_total]);
								fs.vy.push_back(feature.vy[idx_total]);
								fs.vx_dis.push_back(feature.vx_dis[idx_total]);
								fs.vy_dis.push_back(feature.vy_dis[idx_total]);
								//fs.vvDesc.push_back(vFeature[iFeature].vvDesc[idx_total]);
								vector<double> meanDesc;
								if ((fs.vvDesc.size() == 0) || (feature.vvDesc.size() == 0))
								{
									if (fs.vvDesc.size() == 0)
									{
										meanDesc = feature.vvDesc[0];
									}
									else
									{
										meanDesc = fs.vvDesc[0];
									}
								}
								else
								{
									ComputeMeanDescriptor(fs.vvDesc[0], feature.vvDesc[0], meanDesc);
								}							
								fs.vvDesc.clear();
								fs.vvDesc.push_back(meanDesc);
							}
							else if ((it_total1 == feature.vFrame.end()) && (it_total2 != vStitchedFeature[iStitchedFeature].vFrame.end()))
							{
								int idx_total = (int) (it_total2 - vStitchedFeature[iStitchedFeature].vFrame.begin());
								fs.vFrame.push_back(vStitchedFeature[iStitchedFeature].vFrame[idx_total]);
								fs.vCamera.push_back(vStitchedFeature[iStitchedFeature].vCamera[idx_total]);
								fs.vx.push_back(vStitchedFeature[iStitchedFeature].vx[idx_total]);
								fs.vy.push_back(vStitchedFeature[iStitchedFeature].vy[idx_total]);
								fs.vx_dis.push_back(vStitchedFeature[iStitchedFeature].vx_dis[idx_total]);
								fs.vy_dis.push_back(vStitchedFeature[iStitchedFeature].vy_dis[idx_total]);
								//fs.vvDesc.push_back(vStitchedFeature[iStitchedFeature].vvDesc[idx_total]);
								vector<double> meanDesc;
								if ((fs.vvDesc.size() == 0) || (vStitchedFeature[iStitchedFeature].vvDesc.size() == 0))
								{
									if (fs.vvDesc.size() == 0)
									{
										meanDesc = vStitchedFeature[iStitchedFeature].vvDesc[0];
									}
									else
									{
										meanDesc = fs.vvDesc[0];
									}
								}
								else
								{
									ComputeMeanDescriptor(fs.vvDesc[0], vStitchedFeature[iStitchedFeature].vvDesc[0], meanDesc);
								}							
								fs.vvDesc.clear();
								fs.vvDesc.push_back(meanDesc);
							}
							temp = iTotalFrame;
						}

						newFeature = fs;
						feature_idx = iStitchedFeature;
						isInSet = true;
					}
				}
				if (isInSet)
					break;
			}
		}
	}
	if (!isInSet)
	{
		feature_idx = -1;
		newFeature = feature;
	}
}


#pragma once
#include "opencv2/opencv.hpp"
#include <limits>

using namespace cv;
using namespace std;

/*
 GMM - Gaussian Mixture Model
*/
class GMM
{
public:
    static const int componentsCount = 5;
  
    double operator()( const Vec3d color) const;
    double operator()( int ci, const Vec3d color ) const;
	void init( Mat& _model );

    void initLearning();
    void addSample( int ci, const Vec3d color );
    void endLearning();

private:
    void calcInverseCovAndDeterm( int ci );
    Mat model;
    double* coefs;
    double* mean;
    double* cov;

    double inverseCovs[componentsCount][3][3]; 
    double covDeterms[componentsCount];

    double sums[componentsCount][3];
    double prods[componentsCount][3][3];
    int sampleCounts[componentsCount];
    int totalSampleCount;
};
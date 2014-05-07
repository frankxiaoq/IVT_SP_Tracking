#pragma once
#include "opencv2/opencv.hpp"
#include "FgdBgdGMMs.h"
#include "AdaptSkinDetect.h"
#include <time.h>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

const Size WARP_SIZE = Size(24, 24);
const int SIZENUM = WARP_SIZE.width * WARP_SIZE.height * 3;
const int NUM = 600;
const int BATCHSIZE = 5;
const int MAXBASIS = 16;
const int BGBASIS_NUM = 5;						//number of background templates
const float MU = 0.2;							//balance the reconstruction error
const int MAX_LOOP_NUM = 5;						//the maximum number of iterations for L1
const float LAMBDA = 0.05;						//for calculating new likelihood
const double OBJ_THRES = 0.005;					//difference of objective values between 2 steps
const double FORGET_FACTOR = 0.9;				//forgetting factor
const float WEIGHT_THRESHOLD = 0.5;				//threshold value of particles' weight
const double EPSILON = 30.0;					//used to calculate the similarity
const double SIGMA1 = 0.5;						//SIGMA1 & SIGMA2 are threshold of similarity
const double SIGMA2 = 0.8;
const float ALPHA = 0.1;

enum STATUS{
	Detect_Success,
	Tracking_Success,
	Tracking_Unstable,
	Fail
};

typedef struct ResultRect
{
	Point points[4];
}ResultRect;

class LineModTracker
{
public:
	LineModTracker(void);
	~LineModTracker(void);
    bool initialize(Size _image_size, float* _aff_sig, float* _hsv_weight);
	void process(const Mat& image);
	void testBench(const Mat &image, Rect _rect);
	void drawResult(Mat& image, int method);
	inline STATUS getStatus() {return status;}
    inline void setStatus(STATUS sta) {status = sta;}
	ofstream result_file;

private:
	void imgWarp(Mat& src, Mat& dst, Mat& filter, const Size& size=Size(0,0), float scale=1);
	bool tracking(const Mat& image);
	void concatenate(Mat& src1, Mat& src2, Mat& dst, bool r_or_c = false);
	void sklm();
	void calResult(float* est_data, const Point2d* p_points, Point* p_out_points);

	ResultRect palm_result;
	STATUS status;
	Size image_size;
    CascadeClassifier palm_classifier;
	AdaptSkinDetect skin_detecter;

	RNG rng;
	Mat aff_sig;
	Mat est;
	Mat warp_img;
	Mat warp_imgs;
	Mat aver;
	Mat filter;
    Mat hsv_weight;
	SVD svd;
	double n;
	Vec2f velocity;

	bool test_flag;
	FgdBgdGMMs fbGMMs;
	Vec2f pre_wh;
};


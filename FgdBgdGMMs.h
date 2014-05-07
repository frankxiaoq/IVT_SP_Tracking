#pragma once
#include "opencv2/opencv.hpp"
#include <limits>
#include "GMM.h"

using namespace cv;
using namespace std;

class FgdBgdGMMs
{
public:
	FgdBgdGMMs(void);
	~FgdBgdGMMs(void);
	void initGMMs( const Mat& img, Rect box);
	Mat calObjectProImg(const Mat& img);
	
private:
	GMM bgdGMM, fgdGMM;
	Mat bgdModel, fgdModel;
	Mat objectProImg;
};
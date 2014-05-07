#pragma once
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

const static float	 UPSLOPE1 = -1.222222222;
const static float  UPINTCPT1 = 267.33333333;

const static float  UPSLOPE2 = 0.875;
const static float  UPINTCPT2 = 29.375;

const static float  DOWNSLOPE1 = -1.333333333;
const static float  DOWNINTCPT1 = 316.33333333;

const static float  DOWNSLOPE2 = -0.06451612903;
const static float  DOWNINTCPT2 = 170.612903225;

class AdaptSkinDetect
{
public:
	AdaptSkinDetect(void);
	~AdaptSkinDetect(void);
	void skinSegment(const Mat& src, Mat& dst,
		const Rect& obj_mask=Rect(), bool normal=true);
	void adaptBestParam(const Mat& _img, const Rect& adapt_roi);
    inline bool isSkinColor() {return is_skin_color;}

private:
    void setMask(const Mat& _img, const Rect& adapt_roi);
	void skinDetect(const Mat& _img, Mat& skin_img, 
		const float& _upIntcpt1, const float& _upIntcpt2,
		const float& _downIntcpt1, const float& _downIntcpt2);
	float findFitness(const Mat& _img, const float& _upIntcpt1, const float& _upIntcpt2,
		const float& _downIntcpt1, const float& _downIntcpt2);

	float up_intcpt_best1;
	float up_intcpt_best2;
	float down_intcpt_best1;
	float down_intcpt_best2;
    bool is_skin_color;
	Vec<int, 5> skin_val;		//y, cr, cb, cr variance, cb variance
	Mat skin_mask;
};


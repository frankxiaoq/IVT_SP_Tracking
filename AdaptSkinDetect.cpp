#include "AdaptSkinDetect.h"


AdaptSkinDetect::AdaptSkinDetect(void)
{
}


AdaptSkinDetect::~AdaptSkinDetect(void)
{
}

void AdaptSkinDetect::setMask(const Mat& _img, const Rect& adapt_roi)
{
	int num(0);
	vector<Mat> planes;

	skin_mask = Mat::zeros(_img.size(), CV_8UC1);

	for (int y = 0; y < skin_mask.rows; y++)
	{
		uchar* Mi = skin_mask.ptr(y);
		for (int x = 0; x < skin_mask.cols; x++)
		{
			if (x > adapt_roi.x && x < adapt_roi.br().x && y > adapt_roi.y && y < adapt_roi.br().y)
			{
				Mi[x] = 255;
			}
			else
			{
				Mi[x] = 0;
			}
		}
	}

	Mat ycrcb;
	cvtColor(_img, ycrcb, CV_RGB2YCrCb);
	Mat ycrcbROI = ycrcb(adapt_roi);

	split(ycrcbROI, planes);
	MatIterator_<uchar> it_y = planes[0].begin<uchar>(), 
		it_cr = planes[1].begin<uchar>(),
		it_cr_end = planes[1].end<uchar>(),
		it_cb = planes[2].begin<uchar>();
	int y_val(0), cr_val(0), cb_val(0);
	for (; it_cr != it_cr_end; it_y++,it_cr++,it_cb++)
	{
		if (*it_cr > 95 && *it_cr < 125 && *it_cb > 130 && *it_cb < 167)
		{
			y_val += *it_y;
			cr_val += *it_cr;
			cb_val += *it_cb;
			num++;
		}
	}

    if (num == 0)
    {
        skin_val[0] = 0;
        skin_val[1] = 0;
        skin_val[2] = 0;
        skin_val[3] = 0;
        skin_val[4] = 0;
        is_skin_color = false;
    }
    else
    {
        skin_val[0] = y_val / num;
        skin_val[1] = cr_val / num;
        skin_val[2] = cb_val / num;

        planes.clear();
        split(ycrcb, planes);
        MatIterator_<uchar> it_mask = skin_mask.begin<uchar>();
        it_cr = planes[1].begin<uchar>();
        it_cr_end = planes[1].end<uchar>();
        it_cb = planes[2].begin<uchar>();
        for (; it_cr != it_cr_end; it_cr++,it_cb++,it_mask++)
        {
            if (*it_mask != 0)
            {
                if (*it_cr > 95 && *it_cr < 125 && *it_cb > 130 && *it_cb < 167)
                {
                    skin_val[3] += (*it_cr - skin_val[1]) * (*it_cr - skin_val[1]);
                    skin_val[4] += (*it_cb - skin_val[2]) * (*it_cb - skin_val[2]);
                }
                if (abs((*it_cr)-skin_val[1])<9 && abs((*it_cb)-skin_val[2])<9)
                {
                    *it_mask = 255;
                }
                else
                {
                    *it_mask = 0;
                }
            }
        }
        skin_val[3] /= num;
        skin_val[4] /= num;

        is_skin_color = true;
    }
}

void AdaptSkinDetect::skinDetect(const Mat& _img, Mat& skin_img, 
	const float& _upIntcpt1, const float& _upIntcpt2, 
	const float& _downIntcpt1, const float& _downIntcpt2)
{
	if (skin_img.empty() || skin_img.size() != _img.size())
	{
		skin_img = Mat::zeros(_img.size(), CV_8UC1);
	}

	vector<Mat> planes;
	split(_img, planes);
	MatIterator_<uchar> it_b = planes[0].begin<uchar>(),
		it_b_end = planes[0].end<uchar>(),
		it_g = planes[1].begin<uchar>(),
		it_r = planes[2].begin<uchar>(),
		it_bw = skin_img.begin<uchar>();

	float b, g, r, Cb, Cr;
	for (; it_b != it_b_end; ++it_b,++it_g,++it_r,++it_bw)
	{
		b = *it_b;
		g = *it_g;
		r = *it_r;
		Cr = -0.0813*b - 0.4187*g + 0.5*r + 128;
		Cb = 0.5*b - 0.3313*g - 0.1687*r + 128;

		*it_bw = 255* (Cb * UPSLOPE1 + _upIntcpt1 < Cr && 
			Cb * UPSLOPE2 + _upIntcpt2 < Cr && 
			Cb * DOWNSLOPE1 + _downIntcpt1 > Cr && 
			Cb * DOWNSLOPE2 + _downIntcpt2 > Cr);
	}
}

float AdaptSkinDetect::findFitness(const Mat& _img, const float& _upIntcpt1, const float& _upIntcpt2, 
	const float& _downIntcpt1, const float& _downIntcpt2)
{
	Mat skin_img = Mat::zeros(_img.size(), CV_8UC1);
	skinDetect(_img, skin_img, _upIntcpt1, _upIntcpt2, _downIntcpt1, _downIntcpt2);

	int mask_area(0);
	float skin_count(0), bg_count(0);
	uchar* skin_img_data = (uchar*)skin_img.data;
	uchar* mask_data = (uchar*)skin_mask.data;
	for (int i = 0; i < _img.rows*_img.cols; i++,skin_img_data++, mask_data++)
	{
		if (skin_img_data[0] > 0)
		{
			if (mask_data[0] > 0)
			{
				skin_count++;
				mask_area++;
			}
			else
			{
				bg_count++;
			}
		}
		else if (mask_data[0] > 0)
		{
			mask_area++;
		}
	}

	float fit_val = 2 * skin_count / mask_area - bg_count / ((float)(_img.rows*_img.cols-mask_area));

	return fit_val;
}

void AdaptSkinDetect::adaptBestParam(const Mat& _img, const Rect& adapt_roi)
{
	setMask(_img, adapt_roi);

	float best_para(-1000);
	int best_idx(-1000);
	float temp_para;

	Mat mini_img, mini_mask;
	resize(_img, mini_img, Size(160,120));
	resize(skin_mask, skin_mask, Size(160,120));

	//get the best parameter (up_intcpt_best1)
	for (int i = 0; i < 21; i++)
	{
		up_intcpt_best1 = UPINTCPT1 + i - 5;
		temp_para = findFitness(mini_img, up_intcpt_best1, UPINTCPT2, 
			DOWNINTCPT1, DOWNINTCPT2);
		if (best_para < temp_para)
		{
			best_para = temp_para;
			best_idx = i;
		}
	}
	up_intcpt_best1 = UPINTCPT1 + best_idx - 5;

	//get the best parameter (up_intcpt_best2)
	best_para = -1000;
	best_idx = -1000;
	for (int i = 0; i < 21; i++)
	{
		up_intcpt_best2 = UPINTCPT2 + i - 15;
		temp_para = findFitness(mini_img, up_intcpt_best1, up_intcpt_best2, 
			DOWNINTCPT1,DOWNINTCPT2);
		if (best_para < temp_para)
		{
			best_para = temp_para;
			best_idx = i;
		}
	}
	up_intcpt_best2 = UPINTCPT2 + best_idx - 15;

	//get the best parameter (down_intcpt_best1)
	best_para = -1000;
	best_idx = -1000;
	for (int i = 0; i < 21; i++)
	{
		down_intcpt_best1 = DOWNINTCPT1 + i - 15;
		temp_para = findFitness(mini_img, up_intcpt_best1, up_intcpt_best2, 
			down_intcpt_best1, DOWNINTCPT2);
		if (best_para < temp_para)
		{
			best_para = temp_para;
			best_idx = i;
		}
	}
	down_intcpt_best1 = DOWNINTCPT1 + best_idx - 15;

	//get the best parameter (down_intcpt_best2)
	best_para = -1000;
	best_idx = -1000;
	for (int i = 0; i < 21; i++)
	{
		down_intcpt_best2 = DOWNINTCPT2 + i - 15;
		temp_para = findFitness(mini_img, up_intcpt_best1, up_intcpt_best2, 
			down_intcpt_best1, down_intcpt_best2);
		if (best_para < temp_para)
		{
			best_para = temp_para;
			best_idx = i;
		}
	}
    down_intcpt_best2 = DOWNINTCPT2 + best_idx - 15;
}

void AdaptSkinDetect::skinSegment(const Mat& src, Mat& dst, const Rect &obj_mask, bool normal)
{
	if (dst.empty() || dst.size() != src.size())
	{
        if (normal)
            dst = Mat::zeros(src.size(), CV_32FC1);
        else
            dst = Mat::zeros(src.size(), CV_8UC1);
    }

	vector<Mat> planes;
	split(src, planes);
	MatIterator_<uchar> it_b = planes[0].begin<uchar>(),
		it_b_end = planes[0].end<uchar>(),
		it_g = planes[1].begin<uchar>(),
        it_r = planes[2].begin<uchar>();

    if (normal)
    {
        MatIterator_<float> it_bw = dst.begin<float>();
        float b, g, r, Y, Cb, Cr;
        for (int i = 0; it_b != it_b_end; ++it_b,++it_g,++it_r,++it_bw,++i)
        {
            b = *it_b;
            g = *it_g;
            r = *it_r;
            Y = 0.114*b + 0.587*g + 0.299*r;
            Cr = -0.0813*b - 0.4187*g + 0.5*r + 128;
            Cb = 0.5*b - 0.3313*g - 0.1687*r + 128;

            if (Cb * UPSLOPE1 + up_intcpt_best1 < Cr &&
                Cb * UPSLOPE2 + up_intcpt_best2 < Cr &&
                Cb * DOWNSLOPE1 + down_intcpt_best1 > Cr &&
                Cb * DOWNSLOPE2 + down_intcpt_best2 > Cr)
            {
                *it_bw = abs(Y-skin_val[0]) < 15 ? 1.0 : 0.5;
/*
                if ((i%src.cols+1) > obj_mask.x && (i%src.cols+1) < obj_mask.br().x
                        && (i/src.cols+1) > obj_mask.y && (i/src.cols+1) < obj_mask.br().y)
                {   //inside
                    *it_bw = 1.0;
                }
                else    //outside
                {
                    *it_bw = abs(Y-skin_val[0]) < 15 ? 1.0 : 0.5;
                }*/
            }
            else
            {
                *it_bw = 0.1;
            }
        }
    }
    else
    {
        MatIterator_<uchar> it_bw = dst.begin<uchar>();
        float b, g, r, Y, Cb, Cr;
        for (; it_b != it_b_end; ++it_b,++it_g,++it_r,++it_bw)
        {
            b = *it_b;
            g = *it_g;
            r = *it_r;
            Y = 0.114*b + 0.587*g + 0.299*r;
            Cr = -0.0813*b - 0.4187*g + 0.5*r + 128;
            Cb = 0.5*b - 0.3313*g - 0.1687*r + 128;

            if (Cb * UPSLOPE1 + up_intcpt_best1 < Cr &&
                Cb * UPSLOPE2 + up_intcpt_best2 < Cr &&
                Cb * DOWNSLOPE1 + down_intcpt_best1 > Cr &&
                Cb * DOWNSLOPE2 + down_intcpt_best2 > Cr)
            {
                *it_bw = abs(Y-skin_val[0]) < 15 ? 255 : 0;
            }
            else
            {
                *it_bw = 0;
            }
        }
    }
}

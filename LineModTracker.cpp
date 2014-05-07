#include "LineModTracker.h"


LineModTracker::LineModTracker(void)
{
	
}

LineModTracker::~LineModTracker(void)
{

}

bool LineModTracker::initialize(Size _image_size, float* _aff_sig, float *_hsv_weight)
{
	aff_sig = Mat(1, 6, CV_32FC1, _aff_sig).clone();
	rng = RNG(time(NULL));
	est = Mat::zeros(1, 6, CV_32FC1);
    hsv_weight = Mat(1, 3, CV_32FC1, _hsv_weight).clone();
	image_size = _image_size;
	status = Fail;
	n = 0.0;
	velocity[0] = 0.0;
	velocity[1] = 0.0;

    if (!palm_classifier.load("HAAR_FIVE_23_700.xml"))
    {
        cout << "Error!! Can't load the cascade classifier file!" << endl;
        return false;
    }

	test_flag = false;
	result_file.open("ivt_sp.txt", ofstream::out|ofstream::trunc);
	if (!result_file.is_open())
	{
		cout << "Error! Can not open the file!" << endl;
		return false;
	}

	return true;
}

void LineModTracker::process(const Mat& image)
{
	Point2d pt[4] = {Point2d(0,0), Point2d(0,1), Point2d(1,1), Point2d(1,0)};

	if (status == Fail)
	{
        vector<Rect> palms;
        palm_classifier.detectMultiScale(image, palms, 1.1, 3, 0, Size(30,30));

        if (!palms.empty())
		{
            Rect palm = palms[0];

			palm_result.points[0] = Point(palm.x, palm.y);
            palm_result.points[1] = Point(palm.x, palm.y + palm.height);
			palm_result.points[2] = Point(palm.x + palm.width, palm.y + palm.height);
            palm_result.points[3] = Point(palm.x + palm.width, palm.y);

			//for tracking
			float* est_data = (float*)est.data;
            *est_data++ = (float)(palm.x + 0.5*palm.width);
            *est_data++ = (float)(palm.y + 0.55*palm.height);
            *est_data++ = (float)(0.8 * palm.width);
            *est_data++ = (float)(0.75 * palm.height);
			*est_data++ = 0.0;
			*est_data++ = 0.0;

			Mat input_img;
			cvtColor(image, input_img, CV_BGR2HSV);
            input_img.convertTo(input_img, CV_32F, 1/255.0);

			imgWarp(input_img, warp_img, est, WARP_SIZE);
			aver = warp_img.clone();
			warp_imgs = warp_img.clone();
			if (hsv_weight.cols <= 3)
			{
				sqrt(hsv_weight, hsv_weight);
				repeat(hsv_weight, 1, SIZENUM/3, hsv_weight);
			}
            //end for tracking

			status = Detect_Success;
		}
		else
		{
			status = Fail;
		}
	}
	else
    {
        if (tracking(image))
		{
            //calculate the palm result for display
			Point result_points[4];
			float* est_data = (float*)est.data;
            calResult(est_data, pt,result_points);
			palm_result.points[0] = result_points[0];
			palm_result.points[1] = result_points[1];
			palm_result.points[2] = result_points[2];
			palm_result.points[3] = result_points[3];
            status = Tracking_Success;
		}
		else
		{
            status = Fail;
		}
	}
}

void LineModTracker::drawResult(Mat& image, int method)
{
	if (status != Fail)
	{
		//write the result in a file
		result_file << palm_result.points[0].x << " " << palm_result.points[0].y << " " 
			<< palm_result.points[1].x << " " << palm_result.points[1].y << " " 
			<< palm_result.points[2].x << " " << palm_result.points[2].y << " " 
			<< palm_result.points[3].x << " " << palm_result.points[3].y << endl;

		const Point* pts[] = {palm_result.points};
        int points_num = 4;
		switch (method)
		{
        case 0:
            polylines(image, pts, &points_num, 1, true, CV_RGB(0,255,0),3);
			break;
		case 1:
			Point pt;
			pt.x = (palm_result.points[0].x + palm_result.points[1].x 
				+ palm_result.points[2].x + palm_result.points[3].x) / 4;
			pt.y = (palm_result.points[0].y + palm_result.points[1].y 
				+ palm_result.points[2].y + palm_result.points[3].y) / 4;
			float radius;
			radius = sqrtf((palm_result.points[0].x - palm_result.points[1].x) * 
				(palm_result.points[0].x - palm_result.points[1].x) + 
				(palm_result.points[0].y - palm_result.points[1].y) * 
				(palm_result.points[0].y - palm_result.points[1].y)) / 2;
			circle(image, pt, radius, CV_RGB(0, 233, 34), 2);
			break;
		}
    }
}

void LineModTracker::imgWarp(Mat & src,Mat & dst,Mat & filter,const Size & size,float scale)
{
	int i,j,k;
	float cth,sth,cph,sph,ccc,ccs,css,scc,scs,sss;
	int width,height;

	Mat mapx,mapy;
	float * param = (float *)filter.data;
	if(size.width && size.height)
	{
		width = size.width;
		height = size.height;
	}
	else
	{
		if(scale<=0)
		{
			scale = 1;
		}
		width = cvRound(param[2]/scale);
		height = cvRound(param[3]/scale);
	}

	int num = filter.rows;

	mapx.create(num,height*width,CV_32FC1);
	mapy.create(num,height*width,CV_32FC1);

	float * x = (float *)mapx.data;
	float * y = (float *)mapy.data;
	float x_para1,x_para2,x_para3,y_para1,y_para2,y_para3;
	float x_val,y_val;
	float dou_x_para1,dou_x_para2,dou_y_para1,dou_y_para2;
	float w_x,w_y;
	float sq_cph,cp_sp,sq_sph;

	for(k=0;k<num;k++)
	{
		cth = cos(param[4]);
		sth = sin(param[4]);
		cph = cos(param[5]);
		sph = sin(param[5]);

		sq_cph = cph * cph;
		cp_sp = cph * sph;
		sq_sph = sph * sph;

		ccc = cth * sq_cph;
		ccs = cth * cp_sp;
		css = cth * sq_sph;

		scc = sth * sq_cph;
		scs = sth * cp_sp;
		sss = sth * sq_sph;

		x_para1 = (param[2] * (ccc + scs) + param[3] * (css - scs)) / width * 0.5;
		x_para2 = (param[2] * (ccs - scc) - param[3] * (ccs + sss)) / width * 0.5;
		x_para3 = param[0] - width * x_para1 -height * x_para2;


		y_para1 = (param[2] * (scc - ccs) + param[3] * (ccs + sss)) / width * 0.5;
		y_para2 = (param[2] * (ccc + scs) + param[3] * (css - scs)) / width * 0.5;
		y_para3 = param[1] - height * y_para2 - width * y_para1;

		dou_x_para1 = x_para1 + x_para1;
		dou_x_para2 = x_para2 + x_para2;
		x_val = x_para3 - dou_x_para1 - dou_x_para2;


		dou_y_para1 = y_para1 + y_para1;
		dou_y_para2 = y_para2 + y_para2;
		y_val = y_para3 - dou_y_para1 - dou_y_para2;

		w_x = width * dou_x_para1;
		w_y = width * dou_y_para1;

		for (i=0; i < height; i++)
		{
			x_val += dou_x_para2;
			y_val += dou_y_para2;
			for(j=0; j < width; j++)
			{
				x_val += dou_x_para1;
				*x++ = x_val;

				y_val += dou_y_para1;
				*y++ = y_val;
			}
			x_val -= w_x;
			y_val -= w_y;
		}
		param += 6;
	}
    remap(src,dst,mapx,mapy,INTER_NEAREST);
	dst = dst.reshape(1,num);
}

bool LineModTracker::tracking(const Mat& image)
{
	//particle filter
	Mat random(NUM, 6, CV_32FC1);
	rng.fill(random, RNG::NORMAL, Scalar(0), Scalar(1));
	filter = repeat(est, NUM, 1) + random.mul(repeat(aff_sig, NUM, 1));

/*
	Mat random_temp(NUM/6, 6, CV_32FC1);
	rng.fill(random_temp, RNG::NORMAL, Scalar(0), Scalar(1));
	Mat filter_temp;
	Mat est_plus[3];
	est_plus[0] = est.clone();
	est_plus[1] = est.clone();
	est_plus[2] = est.clone();
	est_plus[0].at<float>(0) -= (float)0.2 * est_plus[0].at<float>(2);
	est_plus[0].at<float>(1) -= (float)0.2 * est_plus[0].at<float>(3);
	est_plus[1].at<float>(1) -= (float)0.2 * est_plus[1].at<float>(3);
	est_plus[2].at<float>(0) += (float)0.2 * est_plus[2].at<float>(2);
	est_plus[2].at<float>(1) -= (float)0.2 * est_plus[2].at<float>(3);

	float* fil_data = (float*)filter.data;
	for (int n = 0; n < 3; n++)
	{
		filter_temp = repeat(est_plus[n], NUM/6, 1) + random_temp.mul(repeat(aff_sig, NUM/6, 1));
		float* fil_temp_data = (float*)filter_temp.data;
		for (int i = 0; i < NUM/6; i++)
		{
			for (int j = 0; j < 6; j++)
			{
				*fil_data++ = *fil_temp_data++;
			}
		}
	}*/


	float* fil_data = (float*)filter.data;
	for (int i = 0; i < NUM; i++)
	{
		fil_data[2] = fil_data[2] < WARP_SIZE.width ? WARP_SIZE.width : fil_data[2];
		fil_data[3] = fil_data[3] < WARP_SIZE.height ? WARP_SIZE.height : fil_data[3];
		fil_data[0] = fil_data[0] < 0 ? 0 : fil_data[0];
		fil_data[0] = (fil_data[0]+0.5*fil_data[2] > image_size.width) ?
			(image_size.width-0.5*fil_data[2]) : fil_data[0];
		fil_data[1] = fil_data[1] < 0 ? 0 : fil_data[1];
		fil_data[1] = (fil_data[1]+0.5*fil_data[3] > image_size.height) ?
			(image_size.height-0.5*fil_data[3]) : fil_data[1];

		fil_data += 6;
	}

	//calculate the probability according to the GMMs
	Mat obj_pro_img = fbGMMs.calObjectProImg(image);
	Mat pro_weight = Mat::zeros(1, NUM, CV_32FC1);
	for (int i = 0; i < NUM; ++i)
	{
		float * Mi_filter = filter.ptr<float>(i);
		Rect par_bb;
		par_bb.x = Mi_filter[0] - 0.5*Mi_filter[2];
		par_bb.y = Mi_filter[1] - 0.5*Mi_filter[3];
		par_bb.x = par_bb.x < 0 ? 0 : par_bb.x;
		par_bb.y = par_bb.y < 0 ? 0 : par_bb.y;
		par_bb.x = par_bb.x > image_size.width-1 ? image_size.width-1 : par_bb.x;
		par_bb.y = par_bb.y > image_size.height-1 ? image_size.height-1 : par_bb.y;
		par_bb.width = Mi_filter[2];
		par_bb.height = Mi_filter[3];
		Mat pro_img_tmp = obj_pro_img(par_bb);
		pro_weight.at<float>(i) = sum(pro_img_tmp)[0] / par_bb.area();
	}
	normalize(pro_weight, pro_weight, 1.0, 0.0, NORM_MINMAX, CV_32FC1);
//	double pro_wei_mean = mean(pro_weight)[0];	//mean of probabilities in current frame
	Scalar mean_val, dev_val;
	meanStdDev(pro_weight, mean_val, dev_val);
	double pro_wei_mean = mean_val[0];
	double pro_wei_dev = dev_val[0];

	Mat input_img;
	cvtColor(image, input_img, CV_BGR2HSV);
//	cvtColor(image, input_img, CV_BGR2GRAY);
	input_img.convertTo(input_img, CV_32F, 1/255.0);

	Mat pro_wimgs;
	Mat diff(NUM, SIZENUM, CV_32FC1);
	imgWarp(input_img, pro_wimgs, filter, WARP_SIZE);
	float* diff_data = (float*)diff.data;
	Mat diff_row(1, SIZENUM, CV_32FC1, diff.data);
	for (int i = 0; i < NUM; i++)
	{
		diff.row(i) = (aver - pro_wimgs.row(i)).mul(hsv_weight);
	}

	//build the background dictionary
	float ax = est.at<float>(0) - 1.5 * est.at<float>(2);
	ax = ax < 0 ? 0 : ax;
	float bx = est.at<float>(0) + 1.5 * est.at<float>(2);
	bx = bx > image_size.width ? image_size.width : bx;
	float ay = est.at<float>(1) - 1.5 * est.at<float>(3);
	ay = ay < 0 ? 0 : ay;
	float by = est.at<float>(1) + 1.5 * est.at<float>(3);
	by = by > image_size.height ? image_size.height : by;

	float bb_w = est.at<float>(2);
	float bb_h = est.at<float>(3);
	float bb_x = est.at<float>(0) - 0.5*bb_w;
	float bb_y = est.at<float>(1) - 0.5*bb_h;

	int outlier_num(0);
	Mat est_out(BGBASIS_NUM, 6, CV_32FC1);
	while (outlier_num < est_out.rows)
	{
		float *Mi = est_out.ptr<float>(outlier_num);
		float rx = rng.uniform(ax, bx);
		float ry = rng.uniform(ay, by);
		if (!(rx > bb_x-bb_w && rx < bb_x+bb_w && 
			ry > bb_y-bb_h && ry < bb_y+bb_h))
		{
			Mi[0] = rx + bb_w > image_size.width ? 
				image_size.width - 0.5*bb_w : rx + 0.5*bb_w;
			Mi[1] = ry + bb_h > image_size.height ? 
				image_size.height - 0.5*bb_h : ry + 0.5*bb_h;
			Mi[2] = bb_w;
			Mi[3] = bb_h;
			Mi[4] = est.at<float>(4);
			Mi[5] = est.at<float>(5);
			outlier_num++;
		}
	}
	Mat dict_bg;
	imgWarp(input_img, dict_bg, est_out, WARP_SIZE);
	//end for building the background dictionary

	//calculate likelihood probability
	Mat weight = Mat::zeros(1, NUM, CV_32FC1);
	float* weight_data = (float*)weight.data;
	float weight_sum = 0.0;
	Mat temp_est = Mat::zeros(1, 6, CV_32FC1);						//temp est used to calculate
	Mat temp_img = Mat::zeros(1, pro_wimgs.cols, pro_wimgs.type());	//temp img used to calculate
	float min_err(0.0);
	float err_val(0.0);
	int k(0);
	Mat temp(1, 1, CV_32FC1);
	float* temp_data = (float*)temp.data;
	Mat dict, dict_trans;
	if (!svd.u.empty())		//PCA_L1
	{
		//build the dictionary(target template + background template)
		dict_trans = svd.u.t();
		Mat dict_bg_row = dict_trans.row(dict_trans.rows-1);
		for (int i = 0; i < dict_bg.rows; ++i)	//modified Gram-Schmidt process
		{
			for (int j = i; j < dict_bg.rows; ++j)
			{
				dict_bg.row(j) = dict_bg.row(j) - dict_bg.row(j) * dict_bg_row.t() * dict_bg_row;
			}
			dict_bg.row(i) = dict_bg.row(i) / norm(dict_bg.row(i));
			dict_bg_row = dict_bg.row(i);
			dict_trans.push_back(dict_bg_row);
		}

		dict = dict_trans.t();

		//calculate representation coefficients of all candidates
		Mat coef_basis = Mat::zeros(NUM, dict.cols, CV_32FC1);
		Mat coef_err = Mat::zeros(NUM, SIZENUM, CV_32FC1);
#pragma omp parallel for
		for (int i = 0; i < NUM; ++i)
		{
			Mat coef_basis_row = coef_basis.row(i);		//data shared with the original matrix
			Mat coef_err_row = coef_err.row(i);
			Mat diff_row2 = diff.row(i);
			
			//iterative solution
			float obj_fun_val(0.0);
			for (int j = 0; j < MAX_LOOP_NUM; ++j)
			{
				//fix error, solve basis coefficient
				coef_basis_row = (diff_row2 - coef_err_row) * dict;
				//fix basis, solve error coefficient
				Mat y_m = diff_row2 - coef_basis_row * dict_trans;
				coef_err_row = abs(y_m) - LAMBDA;
				for (int k = 0; k < SIZENUM; ++k)
				{
					coef_err_row.at<float>(k) = coef_err_row.at<float>(k) < 0 ? 
						0 : coef_err_row.at<float>(k);
					coef_err_row.at<float>(k) *= y_m.at<float>(k) < 0 ? -1 : 1;
				}

				Mat obj_fun_tmp = (y_m - coef_err_row) * (y_m - coef_err_row).t();
				obj_fun_tmp += LAMBDA * sum(abs(coef_err_row))[0];
				if (abs(obj_fun_tmp.at<float>(0) - obj_fun_val) < OBJ_THRES)
				{
					break;
				}
				obj_fun_val = obj_fun_tmp.at<float>(0);
			}
		}

		//calculate observation likelihood
		Mat err = diff - coef_err - coef_basis * dict_trans;
		for (int i = 0; i < NUM; ++i)
		{
			float *Mi_err = err.ptr<float>(i);
			float *Mi_coef_err = coef_err.ptr<float>(i);
			float *Mi_coef_basis = coef_basis.ptr<float>(i);

			//background coefficient error 
			float err_coef_bg(0.0);
			for (int j = svd.u.cols; j < coef_basis.cols; ++j)	//just calculate bg part
			{
				err_coef_bg += Mi_coef_basis[j] * Mi_coef_basis[j];
			}

			//reconstruction error
			err_val = 0.0;
			for (int j = 0; j < SIZENUM; ++j)
			{
				err_val += Mi_err[j] * Mi_err[j];
				err_val += (abs(Mi_coef_err[j]) >= LAMBDA) ? LAMBDA : 0;
			}

//			weight_data[i] = exp(-(err_val + MU*err_coef_bg) / EPSILON);
			float sp_weight = exp(-(err_val + MU*err_coef_bg) / EPSILON);
			weight_data[i] = ALPHA * pro_weight.at<float>(i) + (1-ALPHA) * sp_weight;

			if (weight_data[i] >= WEIGHT_THRESHOLD)
			{
				weight_data[i] = pro_weight.at<float>(i) < pro_wei_mean ? 
					ALPHA/2 * pro_weight.at<float>(i) + (1-ALPHA/2) * sp_weight : weight_data[i];
				weight_sum += weight_data[i];
			}
			if ((err_val + MU*err_coef_bg) < min_err || i == 0)
			{
				min_err = (err_val + MU*err_coef_bg);
				k = i;
			}
		}
	}
	else		//traditional PCA
	{
		if (!svd.u.empty())
		{
			Mat uMat, utMat;
			uMat = svd.u;
			utMat = uMat.t();
			float* diff_data = (float*)diff.data;
#pragma omp parallel for
			for (int i = 0; i < NUM; i++)
			{
				float* diffData = diff_data + i * SIZENUM;
				Mat diffRowMat(1, SIZENUM, CV_32FC1, diffData);
				diffRowMat = diffRowMat - diffRowMat * uMat * utMat;
			}
		}
		diff_row.data = diff.data;

		//calculate weight of particles, weight value is 0~1
		size_t step = diff.step;
		for (int i = 0; i < NUM; i++)
		{
			mulTransposed(diff_row, temp, 0);
			err_val = temp_data[0];

			weight_data[i] = exp(-err_val/EPSILON);
			if (weight_data[i] >= WEIGHT_THRESHOLD)
			{
				weight_sum += weight_data[i];
			}

			if (err_val < min_err || i == 0)
			{
				min_err = err_val;
				k = i;
			}
			diff_row.data += step;
		}
	}

    //compare the original est with the warp one*********************
    Mat ori_wimg;
    Mat ori_est = est.clone();
    imgWarp(input_img, ori_wimg, ori_est, WARP_SIZE);
    Mat ori_diff(1, SIZENUM, CV_32FC1);
	ori_diff = aver - ori_wimg;
    ori_diff = ori_diff.mul(hsv_weight);
	float ori_err(0.0);
    if (!svd.u.empty())
    {
		Mat coef_basis = Mat::zeros(1, dict.cols, CV_32FC1);
		Mat coef_err = Mat::zeros(1, SIZENUM, CV_32FC1);
		float obj_fun_val(0.0);
		for (int j = 0; j < MAX_LOOP_NUM; ++j)
		{
			coef_basis = (ori_diff - coef_err) * dict;
			Mat y_m = ori_diff - coef_basis * dict_trans;
			coef_err = abs(y_m) - LAMBDA;
			for (int k = 0; k < SIZENUM; ++k)
			{
				coef_err.at<float>(k) = coef_err.at<float>(k) < 0 ? 
					0 : coef_err.at<float>(k);
				coef_err.at<float>(k) *= y_m.at<float>(k) < 0 ? -1 : 1;
			}

			Mat obj_fun_tmp = (y_m - coef_err) * (y_m - coef_err).t();
			obj_fun_tmp += LAMBDA * sum(abs(coef_err))[0];
			if (abs(obj_fun_tmp.at<float>(0) - obj_fun_val) < OBJ_THRES)
			{
				break;
			}
			obj_fun_val = obj_fun_tmp.at<float>(0);
		}

		Mat err = ori_diff - coef_err - coef_basis * dict_trans;
		float err_coef_bg(0.0);
		for (int j = svd.u.cols; j < coef_basis.cols; ++j)
		{
			err_coef_bg += coef_basis.at<float>(j) * coef_basis.at<float>(j);
		}
		for (int j = 0; j < SIZENUM; ++j)
		{
			ori_err += err.at<float>(j) * err.at<float>(j);
			ori_err += (abs(coef_err.at<float>(j)) >= LAMBDA) ? LAMBDA : 0;
		}
		ori_err += MU * err_coef_bg;
    }
    //***************************************************************

	//select the best candidate as new estimate target object
	Mat pre_img = warp_img.clone();
	Point2f pre_pos(est.at<float>(0), est.at<float>(1));

    if (ori_err < min_err)
    {
        warp_img = ori_wimg.clone();
    }
    else
    {
        for (int i = 0; i < NUM; i++)
        {
            if (weight_data[i] >= WEIGHT_THRESHOLD)
            {
                temp_est += (weight_data[i] / weight_sum) * filter.row(i).clone();
                temp_img += (weight_data[i] / weight_sum) * pro_wimgs.row(i).clone();
            }
        }
        if (temp_est.at<float>(2) == 0 && temp_est.at<float>(3) == 0)
        {
            est = filter.row(k).clone();
            warp_img = pro_wimgs.row(k).clone();
        }
        else
        {
            est = temp_est.clone();
            warp_img = temp_img.clone();
        }
    }

    //calculate the similarity of predict object	p1, p2
	double p1(0.0), p2(0.0);
	Mat diff_simi(1, SIZENUM, CV_32FC1);
	float* diff_simi_data = (float*)diff_simi.data;
	float* curr_data = (float*)warp_img.data;
	float* pre_data = (float*)pre_img.data;
	float* aver_data = (float*)aver.data;

	for (int i = 0; i < SIZENUM; i++)
	{
		*diff_simi_data++ = *curr_data++ - *aver_data++;
	}
	mulTransposed(diff_simi, temp, 0);
	err_val = temp_data[0];
	p1 = exp(-err_val/EPSILON);

	diff_simi_data = (float*)diff_simi.data;
	curr_data = (float*)warp_img.data;
	for (int i = 0; i < SIZENUM; i++)
	{
		*diff_simi_data++ = *curr_data++ - *pre_data++;
	}
	mulTransposed(diff_simi, temp, 0);
	err_val = temp_data[0];
    p2 = exp(-err_val/EPSILON);

	//calculate the velocity of object
	Point2f curr_pos(est.at<float>(0), est.at<float>(1));
	float pre_velo(0.0);
	float curr_velo = sqrt((curr_pos.x - pre_pos.x)*(curr_pos.x - pre_pos.x) + 
		(curr_pos.y - pre_pos.y)*(curr_pos.y - pre_pos.y));
	pre_velo = velocity[0]==0 ? curr_velo : velocity[0];
	velocity[0] = curr_velo;

	float pre_theta(0.0);
	float curr_theta(0.0);
	if (curr_pos.x == pre_pos.x)
	{
		curr_theta = (curr_pos.y - pre_pos.y)>=0 ? 90.0 : 270.0;
	}
	else
	{
		curr_theta = (atan2((curr_pos.y-pre_pos.y), (curr_pos.x-pre_pos.x)) + CV_PI) * 180 / CV_PI;
	}
	pre_theta = velocity[1]==0 ? curr_theta : velocity[1];
	velocity[1] = curr_theta;

	//output the status of tracker, used to choose a proper update method
	int sta(0);
    if ((p1 > SIGMA2 && p2 > SIGMA2) ||
            (p1 > SIGMA1 && p2 > SIGMA2))
	{
		sta = 0;			//no update
	}
    else if (p1 < SIGMA1 && p2 < SIGMA1)
    {
        sta = 2;			//update immediately
    }
    else if ((p1 < SIGMA1 && p2 > SIGMA1))
	{
		sta = 1;			//store image and update
    }

	if (p1 < 0.3 && p2 > SIGMA1)
	{
		sta = 2;
	}

    if (p1 > 0.96 || p1 < 0.01 || (p1 < 0.25 && p2 < 0.01) ||
            (p1 < 0.1 && p2 > 0.95) || p2 > 0.995)
    {
        sta = 3;			//tracking wrong
    }
    else if (abs(curr_theta - pre_theta)>300 && abs(curr_velo-pre_velo)>25)
    {
        sta = 3;			//tracking wrong
    }

//    cout << "p1 p2: " << p1 << " " << p2 << " " << endl;

	switch (sta)
	{
	case 0:
		break;
	case 1:
		concatenate(warp_imgs, warp_img, warp_imgs);
		if (warp_imgs.rows >= BATCHSIZE)
		{
			sklm();
			warp_imgs.release();
		}
		break;
	case 2:
		concatenate(warp_imgs, warp_img, warp_imgs);
		sklm();
		warp_imgs.release();
		break;
	case 3:
		return false;
	}

	return true;
}

void LineModTracker::concatenate(Mat& src1, Mat& src2, Mat& dst, bool r_or_c)
{
	if(src1.empty())
	{
		src2.copyTo(dst);
	}
	else
	{
		CV_Assert((src1.type() == src2.type()) && ((src1.cols == src2.cols) || (src1.rows == src2.rows)));
		int i,j;
		float * src1_data = (float *)src1.data;
		float * src2_data = (float *)src2.data;
		if((src1.cols == src2.cols && src1.rows == src2.rows && r_or_c ==0)
			|| (src1.cols == src2.cols && src1.rows != src2.rows))
		{
			Mat dst1(src1.rows + src2.rows,src2.cols,src2.type());
			float * dst1_data = (float *)dst1.data;
			for(i=0;i<src1.rows;i++)
			{
				for(j=0;j<dst1.cols;j++)
				{
					dst1_data[i*dst1.cols+j] = src1_data[i*dst1.cols+j];
				}
			}
			for(i=0;i<src2.rows;i++)
			{
				for(j=0;j<dst1.cols;j++)
				{
					dst1_data[(i+src1.rows)*dst1.cols+j] = src2_data[i*dst1.cols+j];
				}
			}
			dst1.copyTo(dst);
		}
		else
		{
			Mat dst1(src2.rows,src1.cols+src2.cols,src2.type());
			float * dst1_data = (float *)dst1.data;
			for(i=0;i<src1.rows;i++)
			{
				for(j=0;j<src1.cols;j++)
				{
					dst1_data[i*dst1.cols+j] = src1_data[i*src1.cols+j];
				}
			}
			for(i=0;i<src2.rows;i++)
			{
				for(j=0;j<src2.cols;j++)
				{
					dst1_data[i*dst1.cols+j+src1.cols] = src2_data[i*src2.cols+j];
				}
			}

			dst1.copyTo(dst);
		}
	}
}

void LineModTracker::sklm()
{
	double n0 = n;
	n = (double)warp_imgs.rows;

	if(svd.u.empty())
	{				
		reduce(warp_imgs,aver,0,CV_REDUCE_AVG);
        warp_imgs = (warp_imgs - repeat(aver,warp_imgs.rows,1)).mul(repeat(hsv_weight,warp_imgs.rows,1));
		svd(warp_imgs.t());
	}
	else
	{
		Mat u = svd.u.clone();
		if(!aver.empty())
		{
			Mat aver1;
			reduce(warp_imgs,aver1,0,CV_REDUCE_AVG);
            warp_imgs = (warp_imgs - repeat(aver1,warp_imgs.rows,1)).mul(repeat(hsv_weight,warp_imgs.rows,1));
			Mat addition = sqrt(n*n0/(n+n0)) * (aver - aver1);
			concatenate(warp_imgs,addition,warp_imgs);
			aver = (FORGET_FACTOR * n0 * aver + n * aver1)/(FORGET_FACTOR * n0 + n);
			n = n + FORGET_FACTOR * n0;
		}

		Mat data_proj,data_res;
		Mat wimgs_tran = warp_imgs.t();

		data_proj = svd.u.t() * wimgs_tran;
		data_res = wimgs_tran - svd.u * data_proj;

		Mat qmat,rmat;
		int m = data_res.rows;
		int n = data_res.cols;
		Mat A = data_res.clone();
		qmat.create(m,n,CV_32FC1);
		rmat = Mat::zeros(n,n,CV_32FC1);
		int i,j,k;
		float * r_data = (float*)rmat.data;
		float * q_data = (float *)qmat.data;
		float * A_data = (float *)A.data;
		for(i=0;i<n;i++)
		{
			r_data[i*n+i] = norm(A.col(i));
			if(r_data[i*n+i]!=0)
			{
				qmat.col(i) = A.col(i)/r_data[i*n+i];
			}
			else
			{
				qmat.col(i) = A.col(i)/1;
				qmat.at<float>(i,i) = 1;
			}
			if(i<n-1)
			{
				for(j=i+1;j<n;j++)
				{
					r_data[i*n+j] = qmat.col(i).dot(A.col(j));
					for(k=0;k<m;k++)
					{
						A_data[k*n+j] = A_data[k*n+j] - r_data[i*n+j]*q_data[k*n+i];
					}
				}
			}
		}

		Mat Q;
		concatenate(svd.u,qmat,Q,1);

		Mat R = Mat::diag(svd.w) * FORGET_FACTOR;
		concatenate(R,data_proj,R,1);

		Mat zero = Mat::zeros(wimgs_tran.cols,svd.w.rows,CV_32FC1);

		Mat temp = qmat.t()*data_res;
		concatenate(zero,temp,zero,1);
		concatenate(R,zero,R,0);

		svd(R);

		Mat sqre = svd.w.mul(svd.w);
		double cutoff = sum(sqre).val[0]/1000000;

		Mat keep,new_w,new_u;
		compare(sqre,cutoff,keep,CMP_GE);
		for(i=0;i<keep.rows;i++)
		{
			if((int)(keep.data[i]) != 0)
			{
				Mat num = svd.w.row(i).clone();
				Mat u_col = svd.u.col(i).clone();
				concatenate(new_w,num,new_w);
				concatenate(new_u,u_col,new_u,1);
			}
		}
		new_w.copyTo(svd.w);
		svd.u = Q*new_u;
		int ucols = u.cols;

		if(svd.u.cols > MAXBASIS)
		{
			svd.u = svd.u.colRange(0,MAXBASIS-1).clone();
			svd.w = svd.w.rowRange(0,MAXBASIS-1).clone();
		}
	}
}

void LineModTracker::calResult(float* est_data, const Point2d* p_points, Point* p_out_points)
{
	float rate = est_data[3] / est_data[2];
	float cth, sth, cph, sph, ccc, ccs, css, scc, scs, sss;
	cth = cos(est_data[4]);
	sth = sin(est_data[4]);
	cph = cos(est_data[5]);
	sph = sin(est_data[5]);

	ccc = cth * cph * cph;
	ccs = cth * cph * sph;
	css = cth * sph * sph;

	scc = sth*cph*cph;
	scs = sth*cph*sph;
	sss = sth*sph*sph;

	for(int i = 0; i < 4; i++)
	{
		p_out_points[i].x = est_data[0] + (p_points[i].x-0.5)*((ccc + scs)*est_data[2]+(css - scs)*est_data[3])
			+ (p_points[i].y-0.5)*rate*(est_data[2]*(ccs-scc)+est_data[3]*(-ccs-sss));

		p_out_points[i].y = est_data[1] + (p_points[i].x-0.5)*(est_data[2]*(scc-ccs)+est_data[3]*(ccs+sss))
			+(p_points[i].y-0.5)*rate*(est_data[2]*(ccc+scs)+est_data[3]*(-scs+css));
	}
}

void LineModTracker::testBench( const Mat &image, Rect _rect )
{
	Point2d pt[4] = {Point2d(0,0), Point2d(0,1), Point2d(1,1), Point2d(1,0)};

	if (!test_flag)
	{
		palm_result.points[0] = Point(_rect.x, _rect.y);
		palm_result.points[1] = Point(_rect.x, _rect.y + _rect.height);
		palm_result.points[2] = Point(_rect.x + _rect.width, _rect.y + _rect.height);
		palm_result.points[3] = Point(_rect.x + _rect.width, _rect.y);

		float* est_data = (float*)est.data;
		*est_data++ = (float)(_rect.x + 0.5*_rect.width);
		*est_data++ = (float)(_rect.y + 0.5*_rect.height);
		*est_data++ = (float)_rect.width;
		*est_data++ = (float)_rect.height;
		*est_data++ = 0.0;
		*est_data++ = 0.0;

		//initialize the GMMs
		Rect fb_bb = _rect;
		fb_bb.x += 0.1 * fb_bb.width;
		fb_bb.y += 0.1 * fb_bb.height;
		fb_bb.width *= 0.8;
		fb_bb.height *= 0.8;
		fbGMMs.initGMMs(image, fb_bb);

		Mat input_img;
		cvtColor(image, input_img, CV_BGR2HSV);
//		cvtColor(image, input_img, CV_BGR2GRAY);
		input_img.convertTo(input_img, CV_32F, 1/255.0);
		imgWarp(input_img, warp_img, est, WARP_SIZE);
		aver = warp_img.clone();
		warp_imgs = warp_img.clone();
		if (hsv_weight.cols <= 3)
		{
			sqrt(hsv_weight, hsv_weight);
			repeat(hsv_weight, 1, SIZENUM/3, hsv_weight);
		}

		test_flag = true;
		status = Detect_Success;
	}
	else
	{
		if (tracking(image))
		{
			//calculate the palm result for display
			Point result_points[4];
			float* est_data = (float*)est.data;
			calResult(est_data, pt,result_points);
			palm_result.points[0] = result_points[0];
			palm_result.points[1] = result_points[1];
			palm_result.points[2] = result_points[2];
			palm_result.points[3] = result_points[3];
			status = Tracking_Success;
		}
		else
		{
			status = Fail;
		}
	}
}

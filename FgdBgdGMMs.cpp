#include "FgdBgdGMMs.h"

using namespace cv;

FgdBgdGMMs::FgdBgdGMMs(void)
{
	bgdGMM.init( bgdModel );
	fgdGMM.init( fgdModel );
}

FgdBgdGMMs::~FgdBgdGMMs(void)
{

}

/*
  Initialize GMM background and foreground models using kmeans algorithm.
*/
void FgdBgdGMMs::initGMMs( const Mat& img, Rect box)
{
    const int kMeansItCount = 10;
    const int kMeansType = KMEANS_PP_CENTERS;

    Mat bgdLabels, fgdLabels;
    vector<Vec3f> bgdSamples, fgdSamples;
    Point p;
	
	Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC1 );
    (mask(box)).setTo( Scalar(255) );
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
            if( mask.at<uchar>(p) == 0)
                bgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
            else
                fgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
        }
    }
    CV_Assert( !bgdSamples.empty() && !fgdSamples.empty() );
    Mat _bgdSamples( (int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0] );
    kmeans( _bgdSamples, GMM::componentsCount, bgdLabels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
    Mat _fgdSamples( (int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0] );
    kmeans( _fgdSamples, GMM::componentsCount, fgdLabels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );

    bgdGMM.initLearning();
    for( int i = 0; i < (int)bgdSamples.size(); i++ )
        bgdGMM.addSample( bgdLabels.at<int>(i,0), bgdSamples[i] );
    bgdGMM.endLearning();

    fgdGMM.initLearning();
    for( int i = 0; i < (int)fgdSamples.size(); i++ )
        fgdGMM.addSample( fgdLabels.at<int>(i,0), fgdSamples[i] );
    fgdGMM.endLearning();
}

Mat FgdBgdGMMs::calObjectProImg(const Mat& img)
{
	Point p;
	Mat objectProImg = Mat::zeros(img.rows, img.cols, CV_64FC1);
	Mat fgdProImg = Mat::zeros(img.rows, img.cols, CV_64FC1);
	Mat bgdProImg = Mat::zeros(img.rows, img.cols, CV_64FC1);
	for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++)
        {
            Vec3b color = img.at<Vec3b>(p);
			fgdProImg.at<double>(p) = fgdGMM(color);
			bgdProImg.at<double>(p) = bgdGMM(color);
        }
	}
	normalize(fgdProImg, fgdProImg, 1.0, 0.5, NORM_MINMAX);
	normalize(bgdProImg, bgdProImg, 0.5, 0.0, NORM_MINMAX);
	objectProImg = fgdProImg - bgdProImg;
	normalize(objectProImg, objectProImg, 1.0, 0.0, NORM_MINMAX, CV_32FC1);

	return objectProImg;
}
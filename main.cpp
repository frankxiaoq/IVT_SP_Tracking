#include "LineModTracker.h"

int main()
{
	float aff_palm[] = {3.0, 3.0, 0.01, 0.01, 0.002, 0.001};	//x, y, width, height, rotation, scale for palm
    float hsv_weight[] = {0.2, 0.4, 0.4};       //sum of this three elements must be 1
//	Rect bb(129, 115, 85, 99);	//face
//	Rect bb(226, 92, 43, 45);	//hand
//	Rect bb(205, 115, 33, 103);	//woman
	Rect bb(124, 92, 46, 59);	//cup
//	Rect bb(153, 92, 18, 48);	//bicycle
//	Rect bb(339, 163, 25, 60);	//bolt
//	Rect bb(256, 162, 56, 35);	//car
//	Rect bb(131, 51, 36, 89);	//juice
//	Rect bb(325, 187, 54, 51);	//jump
//	Rect bb(66, 56, 70, 277);	//singer
//	Rect bb(83, 93, 36, 50);	//sunshade
//	Rect bb(167, 80, 49, 50);	//torus
//	Rect bb(323, 229, 59,71);	//palm7
//	Rect bb(170, 40, 62, 82);	//david 390
//	Rect bb(120, 56, 82, 106);	//faceocc2
//	Rect bb(119, 58, 51, 50);	//sylv
//	Rect bb(125, 50, 50, 70);	//girl
	ifstream img_file_name("cup.txt");
	if (!img_file_name.is_open())
	{
		cout << "Error! Can not open the file!" << endl;
	}
	string img_name;

	Mat frame;
/*
	VideoCapture cap("M:\\Visual Studio 2010\\IVT_SP_TESTBENCH\\test resource\\video\\palm7.avi");
	if (!cap.isOpened())
	{
		cout << "ERROR: Can't open the camera!" << endl;
		return -1;
	}
	cap >> frame;*/
	getline(img_file_name, img_name);
	frame = imread(img_name);
//	resize(frame, frame, Size(256,192));

	LineModTracker palm_tracker;
    if (!palm_tracker.initialize(frame.size(), aff_palm, hsv_weight))
    {
        cout << "Palm tracker initialize fail!" << endl;
        return -1;
    }

	while(true)
	{
		Mat show_img;
		double t = (double)cvGetTickCount();
		
//        cap >> frame;
		getline(img_file_name, img_name);
		if (img_name.empty())
		{
			break;
		}
		frame = imread(img_name);
//		resize(frame, frame, Size(256,192));
		if (frame.empty())
		{
			break;
		}
		frame.copyTo(show_img);
		
        //tracking
		palm_tracker.testBench(frame, bb);

        //draw the result
        palm_tracker.drawResult(show_img, 0);

		t = (double)cvGetTickCount() - t;
		cout << "cost time: " << t / ((double)cvGetTickFrequency()*1000.) << endl;

		imshow("frame", show_img);

		if (waitKey(10) == 27)
		{
			break;
        }
	}
	palm_tracker.result_file.close();

	return 0;
}

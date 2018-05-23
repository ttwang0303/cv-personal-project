#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

int main()
{
   CascadeClassifier facecascade;
   facecascade.load("models/lbpcascade_frontalface.xml"); //人脸检测器(快速的LBP)
   VideoCapture capture;
   capture.open(0);
   Mat frame;         //定义一个Mat变量，用于存储每一帧的图像
   int pic_count = 1;
   while(1)
   {
     capture >> frame; //读取当前帧

     std::vector<Rect> faces;
     Mat frameGray;
     cvtColor(frame,frameGray,COLOR_BGR2GRAY);

     facecascade.detectMultiScale(frameGray,faces, 1.1, 4, 0, Size(100, 100), Size(500,500));
     for(size_t i = 0; i < faces.size(); i++)
     {
        rectangle(frame, faces[i], Scalar(0, 255, 0), 2, 8, 0);
     }

     if(faces.size() == 1)
     {
        Mat faceROI = frameGray(faces[0]);
	Mat myFace;
	resize(faceROI, myFace,Size(92,112));
	putText(frame, to_string(pic_count), faces[0].tl(), FONT_HERSHEY_DUPLEX, 1.2, Scalar(71, 99, 255), 2, LINE_8);

	string filename = format("orl_faces/s41/%d.jpg", pic_count);
	imwrite(filename, myFace);
	imshow(filename,myFace);
	waitKey(500);
	destroyWindow(filename);
	pic_count++;
	if(pic_count == 21)
	{
	     return 0;
	}
     }
     imshow("frame",frame);
     waitKey(100);
   }
   return 0;
}
 


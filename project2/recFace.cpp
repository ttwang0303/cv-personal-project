#include "opencv2/opencv.hpp"
#include "opencv2/face.hpp"
#include "opencv2/objdetect.hpp"
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::face;

int main()
{
   VideoCapture cap(0);
   if (!cap.isOpened())
   {
      return -1;
   }
   Mat frame;
   Mat edges;
   Mat gray;

   CascadeClassifier cascade;
   bool stop = false;
   //读取训练好的文件
   cascade.load("models/lbpcascade_frontalface.xml");

   Ptr<BasicFaceRecognizer> myModel = createEigenFaceRecognizer();
   myModel->load("MyEigenFaces.xml");

   while (!stop)
   {
      cap >> frame;

      //建立存放人脸的向量容器
      vector<Rect> faces(0);

      cvtColor(frame, gray, CV_BGR2GRAY);
      //将变换后的图像进行直方图均衡化处理
      equalizeHist(gray, gray);

      cascade.detectMultiScale(gray, faces,

			1.1, 2, 0

			//|CV_HAAR_FIND_BIGGEST_OBJECT  

			//|CV_HAAR_DO_ROUGH_SEARCH  

			| CV_HAAR_SCALE_IMAGE,

			Size(30, 30));

      Mat face;
      Point text_lb;

      for (size_t i = 0; i < faces.size(); i++)
      {
         if (faces[i].height > 0 && faces[i].width > 0)
	 {
	   face = gray(faces[i]);
	   text_lb = Point(faces[i].x, faces[i].y);

	   rectangle(frame, faces[i], Scalar(255,0, 0), 1, 8, 0);
	 }
      }

      Mat face_test;
      int predict = 0;
      if (face.rows >= 120)
      {
        resize(face, face_test, Size(92, 112));
      }
      //Mat face_test_gray;
      //cvtColor(face_test, face_test_gray, CV_BGR2GRAY);
      
      if ( !face_test.empty())
      {
         predict = myModel->predict(face_test);
      }
     
      cout << predict << endl;
      if (predict == 41)
      {
         string name = "WangTan";
         putText(frame, name, text_lb, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
      
      }
     
      imshow("face", frame);
      waitKey(200);
   }
   
   return 0;
}

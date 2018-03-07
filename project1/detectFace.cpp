#include "opencv2/objdetect.hpp"                                                                                   
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Global variables */
String faceCascadePath;          //定义xml文件路径
CascadeClassifier faceCascade;   //定义人脸分类器

int main()
{
  int faceNeighborsMax = 10;
  int neighborStep = 1;
  faceCascadePath = "models/haarcascade_frontalface_default.xml";

  if( !faceCascade.load( faceCascadePath ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
  
  std::vector<Rect> faces;

  Mat frame = imread("1.jpeg");
  Mat frameGray, frameClone;
  cvtColor(frame, frameGray, COLOR_BGR2GRAY);
  
  //-- Detect faces 检测人脸
  for (int neigh = 0; neigh < faceNeighborsMax; neigh = neigh + neighborStep)
  {
    frameClone = frame.clone();
    faceCascade.detectMultiScale( frameGray, faces, 1.2, neigh);

    for ( size_t i = 0; i < faces.size(); i++ )
    {
      int x = faces[i].x;
      int y = faces[i].y;
      int w = faces[i].width;
      int h = faces[i].height;

      rectangle(frameClone, Point(x, y), Point(x + w, y + h), Scalar(255,0,0), 2, 4);      
    }
    putText(frameClone, format("# Neighbors = %d",neigh), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 4);
    imshow( "Face Detection Demo", frameClone );
    int k = waitKey(500);
    if(k == 27)
    {
      destroyAllWindows();
      break;
    }  
  }

}

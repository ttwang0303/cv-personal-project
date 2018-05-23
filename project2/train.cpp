#include "opencv2/face/facerec.hpp"
#include "opencv2/face.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <fstream>                     //文件操作的集合，以流的方式进行
#include <sstream>                     //该库定义了stringstream类，即:流的输入输出操作
#include <math.h>                      //使用string对象代替字符数组，避免缓冲区溢出的危险

using namespace cv;
using namespace cv::face;
using namespace std;

static Mat norm_0_255(InputArray _src)
{
   Mat src = _src.getMat();                //将传入的类型为InputArray的参数转换为Mat的结构
   //创建和返回一个归一化后的图像矩阵:
   Mat dst;
   switch (src.channels())
   {
     case 1:
	   cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
	   break;
     case 3:
	   cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
	   break;
     default:
	   src.copyTo(dst);
	   break;
   }
   return dst;
}

//读取训练图像和标签，使用stringstream和getline方法
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';')
{
	std::ifstream file(filename.c_str(), ifstream::in);//以输入方式打开文件    c_str()函数将字符串转化为字符数组,返回指针
	if(!file)
	{
	  string error_message = "No valid input file was given, please check the given filename.";
	  CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while(getline(file,line))                   //getline{字符数组,字符个数,终止标志符}
	{
	   stringstream liness(line);
	   getline(liness, path, separator);       //遇到分号就结束
	   getline(liness,classlabel);             //继续从分号后边开始，遇到换行结束
	   if (!path.empty() && !classlabel.empty())
	   {
	      images.push_back(imread(path,0));
	      labels.push_back(atoi(classlabel.c_str()));//atoi函数将字符串转换为整数值
	   }

	}
}

int main()
{
  //Get the path to your CSV.
  string fn_csv = "at.txt";
  
  //these vectors hold the images and corresponding labels.
  vector<Mat> images;
  vector<int> labels;

  //Read in the data.this can fail if no valid input filename is given.
  try 
  {
    read_csv(fn_csv, images, labels);
  }
  catch (cv::Exception& e)
  {
    cerr << "Error opening file \"" << fn_csv << "\".Reason: " << e.msg << endl;
    //nothing more we can do
    exit(1);
  }
  // Quit if there are not enough images for this demo.
  if(images.size() <= 1)
  {
    string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
    CV_Error(CV_StsError, error_message);
  }
  
  for (int i = 0; i < images.size(); i++)
  {
       if(images[i].size() != Size(92, 112))
       {
           cout << i << endl;
           cout << images[i].size() << endl;
       }
  }
  //下面这几句是获取数据集的最后一张图片,并将其删除,为了
  //使训练数据和我们测试模型的测试数据不会重叠.
  Mat testSample = images[images.size() - 1];
  int testLabel = labels[labels.size() - 1];
  images.pop_back();    //删除最后一张照片
  labels.pop_back();    //删除最后一个标签

  //下面几行创建用于人脸识别的特征脸模型,并使用从给定
  //CSV文件中读取的图像和标签对其进行训练.
  //这里是一个完整的PCA变换.
  
  //如果你想保留10个主成分,调用
  //这个语句:   cv::createEigenFaceRecognizer(10);
  
  //如果要创建具有置信度阈值(例如123.0)的FaceRecognizer,请使用以下命令
  //调用 :  cv::createEigenFaceRecognizer(10,123.0);

  //若想使用所有特征,并且有一个阈值,那么调用下面这个命令:
  // cv::createEigenFaceRecognizer(0, 123.0);

  Ptr<BasicFaceRecognizer> model0 = createEigenFaceRecognizer();
  model0->train(images,labels);

  //save the model to .xml
  model0->save("MyEigenFaces.xml");

  Ptr<BasicFaceRecognizer> model1 = createFisherFaceRecognizer();
  model1->train(images,labels);
  model1->save("MyFisherFaces.xml");

//  Ptr<BasicFaceRecognizer> model2 = createLBPHFaceRecognizer();
//  model2->train(images,labels);
//  model2->save("MyLBPFaces.xml");

  //紧接着对测试图像进行预测,
  int predictedLabel0 = model0->predict(testSample);
  int predictedLabel1 = model1->predict(testSample);
//  int predictedLabel2 = model2->predict(testSample);
 
  string result_message0 = format("Predicted class = %d / Actual class = %d.",predictedLabel0, testLabel);
  string result_message1 = format("Predicted class = %d / Actual class = %d.",predictedLabel1, testLabel);
//  string result_message2 = format("Predicted class = %d / Actual class = %d.",predictedLabel2, testLabel);

  cout << result_message0 << endl;
  cout << result_message1 << endl;
//  cout << result_message2 << endl;

  waitKey(0);
  return 0;
}

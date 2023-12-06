#include<iostream>
#include<opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/core/core.hpp>  
#include <vector>  
#include <fstream>  
#include <string>  
#include<inttypes.h>
using namespace std;
using namespace cv;

int reverse(int i)
{
	unsigned char c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void extract(const string& test_img_path, const string& test_label_path)
{

	ifstream test_image(test_img_path, ios::in | ios::binary);
	ifstream test_label(test_label_path, ios::in | ios::binary);
	if (test_image.is_open() == false)
	{
		cout << "open mnist image file error!" << endl;
		return;
	}
	if (test_label.is_open() == false)
	{
		cout << "open mnist label file error!" << endl;
		return;
	}

	uint32_t magic;
	uint32_t num_items;
	uint32_t num_label;
	uint32_t rows;
	uint32_t cols;
	//read magic
	test_image.read(reinterpret_cast<char*>(&magic), 4);
	magic = reverse(magic);
	if (magic != 2051)
	{
		cout << "this is not the mnist image file" << endl;
		return;
	}
	test_label.read(reinterpret_cast<char*>(&magic), 4);
	magic = reverse(magic);
	if (magic != 2049)
	{
		cout << "this is not the mnist label file" << endl;
		return;
	}
	//read image/label num
	test_image.read(reinterpret_cast<char*>(&num_items), 4);
	num_items = reverse(num_items);
	test_label.read(reinterpret_cast<char*>(&num_label), 4);
	num_label = reverse(num_label);

	if (num_items != num_label)
	{
		cout << "the image file and label file are not a pair" << endl;
	}

	test_image.read(reinterpret_cast<char*>(&rows), 4);
	rows = reverse(rows);
	test_image.read(reinterpret_cast<char*>(&cols), 4);
	cols = reverse(cols);
	//extract
	for (int i = 0; i != num_items; i++)
	{
		char* pixels = new char[rows * cols];
		test_image.read(pixels, rows * cols);
		char label;
		test_label.read(&label, 1);
		Mat image(rows, cols, CV_8UC1);
		for (int m = 0; m != rows; m++)
		{
			uchar* ptr = image.ptr<uchar>(m);
			for (int n = 0; n != cols; n++)
			{
				if (pixels[m * cols + n] == 0)
					ptr[n] = 0;
				else
					ptr[n] = 255;
			}
		}
		string saveFile = "C:\\Users\\30706\\Desktop\\images\\" + to_string((unsigned int)label) + "_" + to_string(i) + ".jpg";
		imwrite(saveFile, image);
	}
}


int main()
{	
	//extract("C:\\Users\\30706\\Desktop\\opencvtest\\train&test\\t10k-images.idx3-ubyte", "C:\\Users\\30706\\Desktop\\opencvtest\\train&test\\t10k-labels.idx1-ubyte");

	//read a handwritten digit
	Mat image = imread("0.jpg", 0);
	Mat img_show = image.clone();
	//unsigned char to float32
	image.convertTo(image, CV_32F);
	//Normalization
	image = image / 255.0;
	//(1,784)
	image = image.reshape(1, 1);

	//create an ANN
	Ptr<cv::ml::ANN_MLP> ann = ml::StatModel::load<ml::ANN_MLP>("dnn_model.xml");
	//predict
	Mat prediction;
	ann->predict(image, prediction);
	double max_val = 0;
	Point max_point;
	minMaxLoc(prediction, NULL, &max_val, NULL, &max_point);
	int prediction_label = max_point.x;
	cout << "The digit is: " << prediction_label << endl << "Degree of confidence : " << max_val << endl;

	imshow("img", img_show);
	

	
	waitKey(0);
	getchar();
	return 0;
}


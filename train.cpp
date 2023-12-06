#include<iostream>
#include<string>
#include<fstream>
#include<opencv.hpp>
using namespace std;
using namespace cv;

string train_image_path = "C:/Users/30706/Desktop/opencvtest/train&test/train-images.idx3-ubyte";
string train_label_path = "C:/Users/30706/Desktop/opencvtest/train&test/train-labels.idx1-ubyte";
string test_images_path = "C:/Users/30706/Desktop/opencvtest/train&test/t10k-images.idx3-ubyte";
string test_labels_path = "C:/Users/30706/Desktop/opencvtest/train&test/t10k-labels.idx1-ubyte";

//transform to least significant bit first
int reverse(int i)
{
	unsigned char c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

Mat read_mnist_labels(const string file_name)
{
	int magic_number;
	int itemNum;
	Mat labelMat;

	ifstream file(file_name, ios::binary);
	cout << "Reading " << file_name << endl;

	//magic_number and itemNum are at the beginning of the label file, so just read them!
	file.read((char*)&magic_number, sizeof(magic_number));
	file.read((char*)&itemNum, sizeof(itemNum));

	// transform to LSB
	magic_number = reverse(magic_number);
	itemNum = reverse(itemNum);

	cout << "magic number = " << magic_number << endl;
	cout << "item number = " << itemNum << endl;

	//read labels
	labelMat = Mat::zeros(itemNum, 1, CV_32SC1); //32-bit Signed integer with 1 Channel (gray image)
	for (int i = 0; i < itemNum; i++)
	{
		unsigned char tmp = 0;
		file.read((char*)&tmp, sizeof(tmp));
		labelMat.at<int>(i, 0) = static_cast<int>(tmp);
	}
	cout << "Label reading completes" << endl;
	file.close();
	return labelMat;
}

Mat read_mnist_images(const string file_name)
{
	int magic_number = 0;
	int itemNum = 0;
	int rowNum = 0;
	int colNum = 0;
	Mat imageMat;

	ifstream file(file_name, ios::binary);
	cout << "Reading " << file_name << endl;

	file.read((char*)&magic_number, sizeof(magic_number));
	file.read((char*)&itemNum, sizeof(itemNum));
	file.read((char*)&rowNum, sizeof(rowNum));
	file.read((char*)&colNum, sizeof(colNum));

	magic_number = reverse(magic_number);
	itemNum = reverse(itemNum);
	rowNum = reverse(rowNum);
	colNum = reverse(colNum);
	cout << "magic number = " << magic_number << endl;
	cout << "item number = " << itemNum << endl;

	imageMat = Mat::zeros(itemNum, rowNum * colNum, CV_32FC1); //32-bit Float with 1 Channel (gray image)
	for (int i = 0; i < itemNum; i++)
	{
		for (int j = 0; j < rowNum * colNum; j++)
		{
			unsigned char tmp = 0;
			file.read((char*)&tmp, sizeof(tmp));
			imageMat.at<float>(i, j) = float(tmp);
		}
	}

	cout << "Image reading completes" << endl;
	file.close();
	return imageMat;
}

Mat one_hot(Mat label, int num)
{
	//e.g. transform 0 into [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	Mat one_hot_label = Mat::zeros(label.rows, num, CV_32FC1);
	for (int i = 0; i < label.rows; i++)
	{
		int j = label.at<int32_t>(i, 0);
		one_hot_label.at<float>(i, j) = 1.0;
	}
	return one_hot_label;
}

int main()
{
	/*
		prepare training data
	*/

	//read training labels
	Mat train_labels = read_mnist_labels(train_label_path);
	//ann requires label data in one-hot type
	train_labels = one_hot(train_labels, 10);

	//read training images
	Mat train_images = read_mnist_images(train_image_path);
	//Normalizaiton
	train_images = train_images / 255.0;

	//read testing labels
	Mat test_labels = read_mnist_labels(test_labels_path);

	//read testing images
	Mat test_images = read_mnist_images(test_images_path);
	//Normalization
	test_images = test_images / 255.0;


	/*
		build an Ann and train
	*/

	Ptr<ml::ANN_MLP> ann = ml::ANN_MLP::create();
	Mat layerN = (Mat_<int>(1, 5) << 784, 128, 64, 32, 10);
	ann->setLayerSizes(layerN);
	//use backpropagation for gradient descent
	ann->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.001, 0.1);
	//use sigmoid function as activation function
	ann->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
	//criteria for termination
	ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 10, 0.0001));

	//train
	Ptr<ml::TrainData> train_data = ml::TrainData::create(train_images, ml::ROW_SAMPLE, train_labels);
	cout << "Training Start" << endl;
	ann->train(train_data);
	cout << "Training Complete" << endl;


	/*
		test
	*/

	Mat prediction;
	ann->predict(test_images, prediction);

	//compute accuracy
	int correctNum = 0;
	for (int i = 0; i < prediction.rows; i++)
	{
		//get i_th image's prediction
		Mat tmp = prediction.rowRange(i, i + 1);
		double max_value = 0;
		Point max_point;
		// find the max probability
		minMaxLoc(tmp, NULL, &max_value, NULL, &max_point);
		int predict_label = max_point.x;
		int true_label = test_labels.at<int32_t>(i, 0);
		if (predict_label == true_label)
		{
			correctNum++;
		}
	}

	double accuracy = double(correctNum) / double(prediction.rows);
	cout << "The accuracy is: " << accuracy * 100 << "%" << endl;

	//save model
	ann->save("dnn_model.xml");

	getchar();
	return 0;
}
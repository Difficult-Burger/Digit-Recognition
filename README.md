## Handwritten Digit Recognition Project Using OpenCV with DNN in C++
This project aims at utilizing **OpenCV** to perform handwritten digit recognition with the **MNIST** dataset.

MNIST Source: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

**Training Set:**

train-images-idx3-ubyte.gz: training set images (9912422 bytes)

train-labels-idx1-ubyte.gz: training set labels (28881 bytes)

**Testing Set:**

t10k-images-idx3-ubyte.gz: test set images (1648877 bytes)

t10k-labels-idx1-ubyte.gz: test set labels (4542 bytes)

### For the whole document, please see report.pdf.

### For VS 2022 Configuration:

Two different projects should be created respectively for train.cpp and predict.cpp.

1. Create a blank project, and create a blank cpp file in the source folder.
2. Right-click the project "predict" in the "Solution Explorer", click "Properties", and edit the following items:
   - VC++ Directories
     - Add "D:\opencv\build\include;" and "D:\opencv\build\include\opencv2;" to Include Directories.
     - Add "D:\opencv\build\x64\vc16\lib;" to Library Directories, if the version number is vc16 and your device is based on x64 architecture. 
   - Linker
     - Input
       - Add "opencv_world481d.lib;" to Additional Dependencies.
       - **Notice:** The number "481" is only for OpenCV 4.8.1, you should check your own .lib file thourgh the path "...\opencv\build\x64\vc16\lib", if the version number is vc16 and your device is based on x64 architecture. 
4. Copy the code and paste into your blank cpp file.
5. Run.

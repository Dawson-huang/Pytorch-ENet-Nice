#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
void scaleDownSampling(const Mat &src, Mat &dst, double xRatio, double yRatio)
{
    //判断是否为uchar型的像素
    CV_Assert(src.depth() == CV_8U);

    // 计算缩小后图像的大小，取整防止越界
    int rows = static_cast<int>(src.rows * xRatio);
    int cols = static_cast<int>(src.cols * yRatio);

    dst.create(rows, cols, src.type());

    const int channels = src.channels();

    switch (channels)
    {
        case 1: //单通道图像
        {
            uchar *p;
            const uchar *origal;

            for (int i = 0; i < rows; i++){
                p = dst.ptr<uchar>(i);
                //间隔采样取基数行
                int row = static_cast<int>((i + 1) / xRatio + 0.5) - 1;
                origal = src.ptr<uchar>(row);
                for (int j = 0; j < cols; j++){
                    //间隔采样取基数列
                    int col = static_cast<int>((j + 1) / yRatio + 0.5) - 1;
                    p[j] = origal[col];  //把像素值赋给dst
                }
            }
            break;
        }

        case 3://三通道图像
        {
            Vec3b *p;
            const Vec3b *origal;

            for (int i = 0; i < rows; i++) {
                p = dst.ptr<Vec3b>(i);
                int row = static_cast<int>((i + 1) / xRatio + 0.5) - 1;
                origal = src.ptr<Vec3b>(row);
                for (int j = 0; j < cols; j++){
                    int col = static_cast<int>((j + 1) / yRatio + 0.5) - 1;
                    p[j] = origal[col]; //把RGB三个值赋给dst
                }
            }
            break;
        }
    }
}
vector<string> split(const string& s, const string& sep)
{
    vector<string> v; 
    string::size_type pos1, pos2;
    pos2 = s.find(sep);
    pos1 = 0;
    while(string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2-pos1));
         
        pos1 = pos2 + sep.size();
        pos2 = s.find(sep, pos1);
    }
    if(pos1 != s.length())
        v.push_back(s.substr(pos1));
    return v;
}

int main(int argc, char* argv[])
{

    if(argv[1] == "help" || argc < 8){
	cout << "please enter 8 parameter" << endl;
        cout << "argv[1]: Original cityscapes datasets txt file" << endl;
        cout << "argv[2]: Path to Original cityscapes datasets" << endl;
        cout << "argv[3]: New txt file of the resized cityscapes datasets" << endl;
        cout << "argv[4]: Path to the resized cityscapes datasets" << endl;
        cout << "argv[5]: Image x-axis reduction ratio" << endl;
        cout << "argv[6]: Image y-axis reduction ratio" << endl;
        cout << "argv[7]: Label x-axis reduction ratio" << endl;
        cout << "argv[8]: Label y-axis reduction ratio" << endl;
        return 0;
    }
    ofstream writeTxt(argv[3]);
    ifstream readTxt(argv[1]);

    string new_Cityscapes_path = argv[4];
    double image_xRatio = atof(argv[5]);
    double image_yRatio = atof(argv[6]);
    double label_xRatio = atof(argv[7]);
    double label_yRatio = atof(argv[8]);

    string fimage, flabel, Wimage, Wlabel;
    int i = 0;
    while (readTxt >> fimage >> flabel){
        Mat new_image, new_label_image;
        Mat image = imread(argv[2] + fimage);
        Mat label_image = imread(argv[2] + flabel, 0);
        scaleDownSampling(image, new_image, image_xRatio, image_yRatio);
        scaleDownSampling(label_image, new_label_image, label_xRatio, label_yRatio);
        vector<string> iFolder = split(fimage, "/");
        vector<string> lFolder = split(flabel, "/");
        Wimage = new_Cityscapes_path+"/leftImg8bit/"+iFolder[3]+"/"+to_string(i)+".png";
        Wlabel = new_Cityscapes_path+"/gtFine/"+lFolder[3]+"/"+to_string(i)+".png";
        imwrite(Wimage, new_image);
        imwrite(Wlabel, new_label_image);
        writeTxt << "/"+Wimage+" /"+Wlabel+"\n"; 
        i++;
    }
    return 0;
}




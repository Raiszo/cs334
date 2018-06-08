#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv ) {
	cout << "Hello" << endl;
	if ( argc != 2 )
		{
			cout <<"usage: DisplayImage.out <Image_Path>\n" << endl;
			return -1;
			    
		}
	Mat image;
	image = imread( argv[1], CV_LOAD_IMAGE_COLOR );
	if ( !image.data )
		{
			cout << "No image data \n" << endl;
			return -1;
			    
		}
	namedWindow("Display Image", WINDOW_AUTOSIZE );
	imshow("Display Image", image);
	
	waitKey(0);
	return 0;
}

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
using namespace cv;
using namespace std;
int main(int argc, char** argv )
{
	if ( argc != 2 )
		{
			printf("usage: DisplayImage.out <Image_Path>\n");
			return -1;
			    
		}
	Mat image;
	image = imread( argv[1], CV_LOAD_IMAGE_COLOR );
	if ( !image.data )
		{
			printf("No image data \n");
			return -1;
			    
		}
	namedWindow("Display Image", WINDOW_AUTOSIZE );
	imshow("Display Image", image);
	
	waitKey(0);
	return 0;
}
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <stdio.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <iostream>

using namespace cv;
using namespace std;

bool readInput(string, vector<Mat>&, vector<Mat>&, Mat&, Mat&, Size&, float&, int&);
bool readMatricesFromFile(Mat&, Mat&, Mat&, Mat&, Mat&, Mat&);
void calibrateStereoCameras(vector<Mat>, vector<Mat>, Size, float, Mat&, Mat&,
        Mat&, Mat&, Mat&, Mat&);
void undistortAndRectify(Mat, Mat, Mat, Mat, Mat, Mat, Mat, Mat, Mat&, Mat&, Mat&);
void computeDisparityMap(Mat, Mat, Mat&);
void computeDepthMap(Mat, Mat, Mat&);
void viewReconstructedScene(Mat, Mat);
vector<vector<Point2f> > findImagePoints(vector<Mat>, Size);
vector<vector<Point3f> > findObjectPoints(vector <Mat>, Size, float);
void calibrateSingleCamera(vector<vector<Point3f> >&, vector<vector<Point2f> >,
        Size, Mat&, Mat&, vector<Mat>&, vector<Mat>&);
void showPics(vector<Mat>);
double computeReprojectionErrors(const vector<vector<Point3f> >&,
        const vector<vector<Point2f> >&,
        const vector<Mat>&, const vector<Mat>&,
        const Mat&, const Mat&,
        vector<float>&);

int main(int argc, char** argv) {

  vector<Mat> leftCalibrationImages, rightCalibrationImages;
  Mat leftImage, rightImage;
  Size chessboardSize;
  float squareSize;
  int loadMatrices;

  // Reads input data from input file
  string inputFile = argc > 1 ? argv[1] : "input.xml";
  if (!readInput(inputFile, leftCalibrationImages, rightCalibrationImages,
          leftImage, rightImage, chessboardSize, squareSize, loadMatrices))
    return 0;

  Mat cameraMatrix_left = Mat::eye(3, 3, CV_64F);
  Mat cameraMatrix_right = Mat::eye(3, 3, CV_64F);
  Mat distCoeffs_left = Mat::zeros(8, 1, CV_64F);
  Mat distCoeffs_right = Mat::zeros(8, 1, CV_64F);
  Mat R, T;

  // Calibrates the stereo set or reads the previously computed matrices from file
  if (loadMatrices == 0) {
    calibrateStereoCameras(leftCalibrationImages, rightCalibrationImages,
            chessboardSize, squareSize, cameraMatrix_left, cameraMatrix_right,
            distCoeffs_left, distCoeffs_right, R, T);
  } else {
    readMatricesFromFile(R, T, cameraMatrix_left, cameraMatrix_right,
            distCoeffs_left, distCoeffs_right);
  }

  cout << "cameraMatrix_left: " << cameraMatrix_left << endl;
  cout << "cameraMatrix_right: " << cameraMatrix_right << endl;
  cout << "distCoeffs_left: " << distCoeffs_left << endl;
  cout << "distCoeffs_right: " << distCoeffs_right << endl;

  // Undistorts and rectifies the two input pictures
  Mat undistortedRectified_left, undistortedRectified_right, Q;
  undistortAndRectify(leftImage, rightImage, cameraMatrix_left,
          cameraMatrix_right, distCoeffs_left, distCoeffs_right, R, T,
          undistortedRectified_left, undistortedRectified_right, Q);

  // Computes the disparity map
  Mat disparity8;
  computeDisparityMap(undistortedRectified_left, undistortedRectified_right,
          disparity8);

  // Computes the depth map
  Mat depthMap;
  computeDepthMap(disparity8, Q, depthMap);

  // Reconstructs and views the pointcloud
  viewReconstructedScene(undistortedRectified_left, depthMap);

  return 0;
}

// Reads input values from the XML input file.
// Returns true if data could successfully be read.
// The user must take care not to modify the names of the XML fields.
bool readInput(string fileName, vector<Mat>& leftCalibrationImages,
        vector<Mat>& rightCalibrationImages, Mat& leftImage, Mat& rightImage,
        Size& chessboardSize, float& squareSize, int& loadMatrices) {

  FileStorage fs(fileName, FileStorage::READ);

  if (!fs.isOpened()) {
    cout << "Failed to open file" << endl;
    return false;
  }

  // Number of inner chessboard corners per row/column
  int rows, cols;
  fs["rows"] >> rows;
  fs["columns"] >> cols;
  chessboardSize = Size(cols, rows);

  // Defines whether calibration matrices should be loaded from an XML file
  fs["load-matrices"] >> loadMatrices;

  // Length of a chessboard square
  fs["square-size"] >> squareSize;

  // Loads calibration images taken by left camera
  FileNode images = fs["calibration-images-left"];
  FileNodeIterator it = images.begin(), it_end = images.end();
  int i = 0;
  for (; it != it_end; ++it, i++) {
    leftCalibrationImages.push_back(imread((string) * it));
  }

  // Loads calibration images taken by right camera
  images = fs["calibration-images-right"];
  it = images.begin(), it_end = images.end();
  i = 0;
  for (; it != it_end; ++it, i++) {
    rightCalibrationImages.push_back(imread((string) * it));
  }

  // Loads left and right input images
  string imagePath;
  fs["input-image-left"] >> imagePath;
  leftImage = imread(imagePath);
  fs["input-image-right"] >> imagePath;
  rightImage = imread(imagePath);

  fs.release();

  return true;
}

// Reads calibration matrices from matrices.xml.
// Returns true if data could successfully be read.
bool readMatricesFromFile(Mat& R, Mat& T,
        Mat& cameraMatrix_left, Mat& cameraMatrix_right, Mat& distCoeffs_left,
        Mat& distCoeffs_right) {
  FileStorage fs("matrices.xml", FileStorage::READ);

  if (!fs.isOpened()) {
    cout << "Failed to open file" << endl;
    return false;
  }

  fs["R"] >> R;
  fs["T"] >> T;
  fs["left-camera-matrix"] >> cameraMatrix_left;
  fs["right-camera-matrix"] >> cameraMatrix_right;
  fs["left-dist-coeffs"] >> distCoeffs_left;
  fs["right-dist-coeffs"] >> distCoeffs_right;

  fs.release();

  cout << "Matrices successfully read from file" << endl;

  return true;
}

// Views each of the input pictures in a window.
void showPics(vector<Mat> input) {
  for (int i = 0; i < input.size(); i++) {
    namedWindow("Image", WINDOW_NORMAL);
    imshow("Image", input.at(i));
    waitKey(0);
  }
}

// Finds the image points in each of the calibration images.
vector<vector<Point2f> > findImagePoints(vector<Mat> input, Size patternSize) {
  const unsigned char IMG_NUMBER = input.size();

  Mat gray;
  vector<vector<Point2f> > corners(IMG_NUMBER);

  for (int i = 0; i < IMG_NUMBER; i++) {
    cvtColor(input.at(i), gray, CV_RGB2GRAY);

    // Determines whether there is a chessboard pattern in the image and,
    // if present, the location of the internal chessboard corners
    cout << "Searching corners for image #" << i << endl;
    bool patternFound = findChessboardCorners(gray, patternSize, corners.at(i),
            CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
            + CALIB_CB_FAST_CHECK);

    // If a corner pattern is found, computes corner locations with
    // increased (subpixel) accuracy
    if (patternFound) {
      cout << "Corners found. Polishing corners..." << endl << endl;
      cornerSubPix(gray, corners.at(i), Size(11, 11), Size(-1, -1),
              TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
    } else {
      cout << "Corners not found." << endl << endl;
      corners.pop_back();
    }
    /*if(patternFound) {
      drawChessboardCorners(input.at(i), patternSize, Mat(corners.at(i)),
            patternFound);

      namedWindow("Corners", WINDOW_NORMAL);
      imshow("Corners", input.at(i));
      waitKey(0);
    }*/
  }

  return corners;
}

// Initializes the object points - the coordinates of each corner of the
// chessboard in the calibration pattern coordinate space. Returns a vector of
// vectors of points, one for each input image.
// This function works under the convenient assumption that the chessboard was
// kept still and the camera moved while capturing the scene.
// This means that the object points are the same for each of the input
// calibration images.
vector<vector<Point3f> > findObjectPoints(vector<Mat> input, Size patternSize,
        float squareSize) {
  const unsigned char IMG_NUMBER = input.size();


  // Computes the object points
  vector<Point3f> objectPoints;
  for (int i = 0; i < patternSize.height; i++) {
    for (int j = 0; j < patternSize.width; j++) {
      objectPoints.push_back(Point3f(float(j * squareSize), float(i * squareSize), 0));
    }
  }

  vector<vector<Point3f> > output;
  for (int i = 0; i < IMG_NUMBER; i++) {
    output.push_back(objectPoints);
  }

  return output;
}

// Calibrates a single camera, computing the intrinsic matrix and the distortion
// coefficients matrix.
void calibrateSingleCamera(vector<vector<Point3f> >& objectPoints,
        vector<vector<Point2f> > imagePoints, Size imageSize, Mat& cameraMatrix,
        Mat& distCoeffs, vector<Mat>& rvecs, vector<Mat>& tvecs) {

  // Ensures the objectPoints vector of vectors is of the same size as the
  // imagePoints one, and fills it with objectPoints
  objectPoints.resize(imagePoints.size(), objectPoints.at(0));

  // Calibrates the camera
  bool successful = calibrateCamera(objectPoints, imagePoints, imageSize,
          cameraMatrix, distCoeffs, rvecs, tvecs);

  if (!successful) {
    cout << "Couldn't calibrate camera" << endl;
    return;
  }

  cout << "Camera successfully calibrated" << endl;

  // Computes the average reprojection error of the calibration
  vector<float> perViewErrors;
  double rms = computeReprojectionErrors(objectPoints, imagePoints, rvecs,
          tvecs, cameraMatrix, distCoeffs, perViewErrors);
  cout << "rms: " << rms << endl;
  /*for (int i = 0; i < perViewErrors.size(); i++) {
    cout << "perViewErrors for image #" << i << ": " << perViewErrors.at(i) << endl;
  }*/

}

// Calibrates the stereo camera pair, computing the intrinsic and distortion
// matrices of each camera as well as the rotation matrix and translation vector
// between the two cameras' coordinate systems.
void calibrateStereoCameras(vector<Mat> leftImages, vector<Mat> rightImages,
        Size chessboardSize, float squareSize, Mat& cameraMatrix_left,
        Mat& cameraMatrix_right, Mat& distCoeffs_left, Mat& distCoeffs_right,
        Mat& R, Mat& T) {

  // Computes image points for each camera
  vector<vector<Point2f> > imagePoints_right =
          findImagePoints(rightImages, chessboardSize);
  vector<vector<Point2f> > imagePoints_left =
          findImagePoints(leftImages, chessboardSize);

  // Computes object points
  vector<vector<Point3f> > objectPoints = findObjectPoints(rightImages,
          chessboardSize, squareSize);
  Size size = leftImages.at(0).size();

  // Calibrates the two cameras separately
  vector<Mat> rvecs, tvecs;
  calibrateSingleCamera(objectPoints, imagePoints_right, size,
          cameraMatrix_right, distCoeffs_right, rvecs, tvecs);
  rvecs.clear();
  tvecs.clear();
  calibrateSingleCamera(objectPoints, imagePoints_left, size, cameraMatrix_left,
          distCoeffs_left, rvecs, tvecs);

  // Calibrates the stereo set, calculating the reprojection error
  Mat E, F;
  double rms = stereoCalibrate(objectPoints, imagePoints_left, imagePoints_right,
          cameraMatrix_left, distCoeffs_left, cameraMatrix_right,
          distCoeffs_right, size, R, T, E, F);
  cout << "Stereo rms: " << rms << endl;

  // Saves the matrices in a file
  FileStorage fs("matrices.xml", FileStorage::WRITE);
  fs << "R" << R;
  fs << "T" << T;
  fs << "left-camera-matrix" << cameraMatrix_left;
  fs << "right-camera-matrix" << cameraMatrix_right;
  fs << "left-dist-coeffs" << distCoeffs_left;
  fs << "right-dist-coeffs" << distCoeffs_right;
  fs.release();
}

// Undistorts and rectifies the input images, one for each camera.
void undistortAndRectify(Mat leftImage, Mat rightImage, Mat cameraMatrix_left,
        Mat cameraMatrix_right, Mat distCoeffs_left, Mat distCoeffs_right,
        Mat R, Mat T, Mat& undistortedRectified_left,
        Mat& undistortedRectified_right, Mat& Q) {

  // Computes rectification transforms, projection matrices and
  // disparity-to-depth mapping matrixfor both cameras
  Size size = leftImage.size();
  Mat R1, R2, P1, P2;
  stereoRectify(cameraMatrix_left, distCoeffs_left, cameraMatrix_right,
          distCoeffs_right, size, R, T, R1, R2, P1, P2, Q);

  // Computes the undistortion and rectification transformation maps for
  // each camera
  Mat leftMap1, leftMap2, rightMap1, rightMap2;
  initUndistortRectifyMap(cameraMatrix_left, distCoeffs_left, R1, P1, size,
          CV_32FC1, leftMap1, leftMap2);
  initUndistortRectifyMap(cameraMatrix_right, distCoeffs_right, R2, P2, size,
          CV_32FC1, rightMap1, rightMap2);

  // Applies the undistortion and rectification transformation maps to the
  // input images
  remap(leftImage, undistortedRectified_left,
          leftMap1, leftMap2, INTER_LINEAR);
  remap(rightImage, undistortedRectified_right,
          rightMap1, rightMap2, INTER_LINEAR);
}

// Computes the disparity map from the undistorted and rectified input images,
// one for each camera.
void computeDisparityMap(Mat undistortedRectified_left,
        Mat undistortedRectified_right, Mat& disparity8) {

  // Converts the input images to grayscale
  Mat gray_left, gray_right, disparity;
  cvtColor(undistortedRectified_left, gray_left, CV_BGR2GRAY);
  cvtColor(undistortedRectified_right, gray_right, CV_BGR2GRAY);

  // Initializes the object used to compute the stereo correspondence via the
  // semi-global block matching algorithm
  //StereoSGBM stereo = StereoSGBM(-64,192,5);
  //StereoSGBM stereo = StereoSGBM(0,272,3,216,864,10,4,1,100,2,false);
  //StereoSGBM stereo = StereoSGBM(0, 272, 3,216,864);
  StereoSGBM stereo = StereoSGBM();
  stereo.minDisparity = 0;
  //stereo.numberOfDisparities = 272;
  stereo.numberOfDisparities = 288;
  stereo.SADWindowSize = 3;
  stereo.P1 = 432; //8*number of image channels*SADWindowSize*SADWindowSize*2
  stereo.P2 = 1928; //32*number of image channels*SADWindowSize*SADWindowSize*2
  //stereo.P1 = 600;
  //stereo.P2 = 3600;
  stereo.disp12MaxDiff = 12;

  // Computes the disparity map
  stereo.operator()(gray_left, gray_right, disparity);

  // Normalizes the disparity map
  normalize(disparity, disparity8, 0, 255, CV_MINMAX, CV_8U);

  // Shows the disparity map
  namedWindow("Disparity map", WINDOW_NORMAL);
  imshow("Disparity map", disparity8);
  waitKey(0);

  //char temp[200];
  //sprintf(temp, "../../../disparity %d,%d,%d.png", n1, n2, n3);
  //imwrite(temp, disparity8);
}

// Computes the depth map from the disparity map and the disparity-to-depth
// mapping matrix.
void computeDepthMap(Mat disparity8, Mat Q, Mat& depthMap) {

  // Computes the depth map
  reprojectImageTo3D(disparity8, depthMap, Q);

  // Shows the depth map
  namedWindow("Depth map", WINDOW_NORMAL);
  imshow("Depth map", depthMap);
  waitKey(0);
}

// Reconstructs and shows the 3D scene in the form of a point cloud from an
// undistorted rectified image and a depth map.

void viewReconstructedScene(Mat image, Mat depthMap) {

  // Threshold value for x/y/z coordinates of the points in the pointcloud
  const short THRESHOLD = 5000;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr output(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointXYZRGB outputPoint;

  // Builds the pointcloud
  for (int i = 0; i < depthMap.rows; i++) {
    for (int j = 0; j < depthMap.cols; j++) {
      Point3f point = depthMap.at<Point3f>(i, j);
      //if (disparity.at<uchar>(i, j) == 0) continue;
      Vec3b pointColor = image.at<Vec3b>(i, j); //BGR 

      // Sets RGB values and XYZ coordinates of the point
      outputPoint.r = pointColor[2];
      outputPoint.g = pointColor[1];
      outputPoint.b = pointColor[0];
      outputPoint.x = point.x;
      outputPoint.y = -point.y;
      outputPoint.z = point.z;

      // If the XYZ absolute values aren't higher than the threshold,
      // the point is pushed into the pointcloud
      if (!(outputPoint.x > abs(THRESHOLD) || outputPoint.y > abs(THRESHOLD)
              || outputPoint.z > abs(THRESHOLD)))
        output->points.push_back(outputPoint);
    }
  }

  // Initializes the point cloud viewer and views the point cloud
  pcl::visualization::PCLVisualizer viewer("Reconstructed scene");

  viewer.setBackgroundColor(0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(output);
  viewer.addPointCloud<pcl::PointXYZRGB> (output, rgb, "cloud");
  viewer.addCoordinateSystem(1.0, "cloud");
  viewer.initCameraParameters();
  viewer.spin();
}

// Computes and returns the average reprojection error of a calibrated camera.
// It should be as close to zero as possible.
double computeReprojectionErrors(const vector<vector<Point3f> >& objectPoints,
        const vector<vector<Point2f> >& imagePoints,
        const vector<Mat>& rvecs, const vector<Mat>& tvecs,
        const Mat& cameraMatrix, const Mat& distCoeffs,
        vector<float>& perViewErrors) {

  vector<Point2f> imagePoints2;
  int i, totalPoints = 0;
  double totalErr = 0, err;
  perViewErrors.resize(objectPoints.size());
  for (i = 0; i < (int) objectPoints.size(); ++i) {
    projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix, // project
            distCoeffs, imagePoints2);
    err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2); // difference
    int n = (int) objectPoints[i].size();
    perViewErrors[i] = (float) std::sqrt(err * err / n); // save for this view
    totalErr += err*err; // sum it up
    totalPoints += n;
  }

  return sqrt(totalErr / totalPoints); // calculate the arithmetical mean
}

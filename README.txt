This program reconstructs the 3D scene, in the form of a pointcloud, captured by two images, one for each head of a stereo camera. If calibration matrices are not already known, the program can calibrate the stereo set with the help of calibration images (for each camera) provided by the user.

Compiling 3DReconstruction:

1. open the terminal in the folder where main.cpp and CMakeLists.txt are located;
2. type "cmake ." (without quotation marks) and press enter;
3. type "make" (without quotation marks) and press enter. The executable will be in the newly created "bin" folder.

4 (FOR THE TEACHERS WHO WILL BE GRADING THIS). place the "left" and "right" folders with the calibration images and the input images in the "bin" folder, the input file is already set up to find everything needed there. This is going to save you a little time as you will not have to modify the calibration paths in input.xml before testing the program with the images you provided.


Usage of 3DReconstruction:

1. program input is done via an XML file (input.xml as default - you can also make your own, syntax must be consistent with input.xml). Refer to the example input file for additional information as to how to configure the input file;
2. launch the program, specifying the path to the XML file as an input parameter (e.g.: ./3DReconstruction my_input_file.xml);
3. after calibrating the cameras for the first time, the program saves the relevant matrices in matrices.xml. Setting the load-matrices parameter != 0 in the input file from step 1 enables future executions of the program to skip the calibration process by reading previously computed calibration matrices from matrices.xml.

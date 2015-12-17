Usage of 3DReconstruction:

1. program input is done via an XML file (input.xml as default - you can also make your own, syntax must be consistent with input.xml). Refer to the 		 example input file for additional information as to how to configure the input file;
2. launch the program, specifying the path to the XML file as an input parameter (e.g.: ./3DReconstruction my_input_file.xml);
3. after calibrating the cameras for the first time, the program saves the relevant matrices in matrices.xml. Setting the load-matrices parameter != 0 	  in the input file from step 1 enables future executions of the program to skip the calibration process by reading previously computed calibration 	 	  matrices from said file.

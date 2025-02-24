Abstract  
The creation and assessment of a deep learning model for graffiti detection in photos and real-time video data using YOLO v5 is presented in this portfolio contribution. The project exhibits a thorough comprehension of PyTorch setup, iterative optimization, evaluation metrics, model training, and data preprocessing. The source code, model results, and labeled datasets are arranged as follows in the submission:  
•	Annotation Conversion Function: To convert the supplied annotation format in training labels to the YOLO annotation format, a new function called convert_annotations is built. This feature makes sure that bounding boxes are formatted and normalized correctly, which makes it easier to integrate them with the YOLO v5 training process.  
•	YOLO v5 Model Training: 400 randomly chosen photos from the training dataset are used to train the YOLO v5 model. The model can correctly identify graffiti in photos thanks to this training procedure, which makes use of the transformed annotations. The model's performance is improved with each iteration of the training process. Every iteration's trained models are saved as best.the week-06-portfolio/train/runs/train/graffiti_detection_iter_X/weights/ folders, where X is the number of iterations.   
•	Iterative Training and Optimization: The YOLO v5 model is retrained using fresh sets of 400 training and 40 test photographs in each iteration of an iterative training procedure. This procedure keeps on until either all of the test images have been used for training and testing, or 80% of the test images have an IoU above 90%. The model from the previous stage serves as the pretrained model for each iteration, promoting progressive learning and performance improvement. Under the train directory, the results of each iteration—such as CSV files and example annotated images—are arranged in the appropriate iteration directories.  
•	Real-Time Video Detection: Graffiti in real-time video footage is detected using the final, optimized YOLO v5 model. After processing a variety of video inputs, the model recognizes and labels occurrences of graffiti with confidence ratings and bounding boxes. The model's real-time detection skills are illustrated using sample video sources from Pexels. The detection results are stored in the results directory, where they are arranged according to each video track in subfolders like track, track2, etc.  
  
  
Repository Structure and Access  
All project requirements, documentation, source code, and results are organized within the week-06-portfolio repository on GitHub. The following links provide access to each component:  

Requirements: https://github.com/dinhdung6/Yolo-graffiti-detection/blob/main/week-06-portfolio/Portfolio-week6.pdf   

Documentation: https://github.com/dinhdung6/Yolo-graffiti-detection/tree/main/week-06-portfolio   

Source Code: https://github.com/dinhdung6/Yolo-graffiti-detection/tree/main/week-06-portfolio/code   

YAML Config: https://github.com/dinhdung6/Yolo-graffiti-detection/tree/main/week-06-portfolio/train/yaml   

YOLO v5 Model Training and Results (train): https://github.com/dinhdung6/Yolo-graffiti-detection/tree/main/week-06-portfolio/train   

Evaluation Images: https://github.com/dinhdung6/Yolo-graffiti-detection/tree/main/week-06-portfolio/train/evaluation_images_iter_{1-30)   

Evaluation Results CSV: https://github.com/dinhdung6/Yolo-graffiti-detection/blob/main/week-06-portfolio/train/evaluation_results_iter_{1-30}.csv   

YOLO v5 Best Model on each Iteration: https://github.com/dinhdung6/Yolo-graffiti-detection/tree/main/week-06-portfolio/train/runs/train/graffiti_detection_iter_{1-30}/weights    
Detection Results (results): https://github.com/dinhdung6/Yolo-graffiti-detection/tree/main/week-06-portfolio/results   

dataset: https://zenodo.org/records/3238357  

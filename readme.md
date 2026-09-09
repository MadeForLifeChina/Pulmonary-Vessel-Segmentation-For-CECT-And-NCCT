🫁 Welcome to the Pulmonary-Vessel-Segmentation-For-CECT-And-NCCT

📌 To develop a deep learning model for pulmonary vessels segmentation on both computed tomography pulmonary angiography (CTPA) and Non-Contrast Computed Tomography (NCCT) data and validate its performance using an external clinical dataset. In this retrospective study, we collected 427 cases of chest CT data from different countries and manufacturers, including 213 CTPA cases and 214 NCCT cases. Then we designed a CTPA-NCCT ground truth (GT) generation method and resulting in a high-precision dataset. Then we developed a 3D UNet based deep learning method with the vessel lumen structure optimization module (VLSOM). We evaluated the results using metrics of Cl-Recall (Centerline Recall) and Cl-DICE (Centerline DICE) focusing on the integrity of vascular structures. Finally, the model performance was visually assessed in an external dataset. The pulmonary vessel segmentation algorithm we designed can improve the connectivity and integrity of the pulmonary vascular structure and can support both CTPA and NCCT data well.

📌For more information, please refer to the following preprint information:https://doi.org/10.48550/arXiv.2503.16988

1.In the loss function section, we designed a hybrid loss function combining DICE Loss, CE Loss, and Centerline DICE Loss (Cl-DICE Loss), and encapsulated it into the file Pulmonary-Vessel-Segmentation-For-CECT-And-NCCT\nnunetv2\training\loss\vessel_loss.py.

2.We used the vessel_loss in Pulmonary-Vessel-Segmentation-For-CECT-And-NCCT\nnunetv2\training\nnUNetTrainer\nnUNetTrainer.py and performed model training.

📌Quick Start

1.Save the images and ground truth (GT) using the default nnU-Net v2 directory structure. Then, run the nnU-Net v2 default plan_and_preprocess command for data preprocessing. Artery label: 1; Vein label: 2;

2.Process the .npz files in nnUNet_preprocessed/Datasetxxx/nnUNetPlans_3d_fullres offline.Then extract the arterial and venous centerlines using standard libraries like scikit-image. Save the extracted centerlines as Label 3 and Label 4, respectively.

3.Once the above steps are completed, simply follow the nnU-Net v2 instructions to train and validate the model.



# Welcome to the Pulmonary-Vessel-Segmentation-For-CECT-And-NCCT

1.In the loss function section, we designed a hybrid loss function combining DICE Loss, CE Loss, and Centerline DICE Loss (Cl-DICE Loss), and encapsulated it into the file Pulmonary-Vessel-Segmentation-For-CECT-And-NCCT\nnunetv2\training\loss\vessel_loss.py.

2.We used the vessel_loss in Pulmonary-Vessel-Segmentation-For-CECT-And-NCCT\nnunetv2\training\nnUNetTrainer\nnUNetTrainer.py and performed model training.

3.lease note that during the training of this network, it is necessary to manually extract the centerlines of the pulmonary arteries and veins offline, and properly configure them in the ground truth (GT). Specifically: Artery label: 1; Vein label: 2; Artery centerline label: 3; Vein centerline label: 4



DATA_FOLDER=/data/dataset/nuc_segmentation/
GPUS=1
create_image:
	sudo bash environment/create-img.sh

benchmark: src/prepare_train_validation_test.nf
	nextflow $< --tnbc_path $(DATA_FOLDER)/TNBC_NucleiSegmentation \
				--monuseg_path $(DATA_FOLDER)/MoNucleiSegmentation \
				--monusac_path $(DATA_FOLDER)/MoNuSAC_images_and_annotations \
				--consep_path $(DATA_FOLDER)/consep/CoNSeP \
				--cpm_path $(DATA_FOLDER)/DataCPM \
				--pan_crops $(DATA_FOLDER)/pan_crops \
				--benchmark 1 \
				-resume

Unet_vgg: src/prepare_train_validation_test.nf
	nextflow $< --tnbc_path $(DATA_FOLDER)/TNBC_NucleiSegmentation \
				--monuseg_path $(DATA_FOLDER)/MoNucleiSegmentation \
				--monusac_path $(DATA_FOLDER)/MoNuSAC_images_and_annotations \
				--consep_path $(DATA_FOLDER)/consep/CoNSeP \
				--cpm_path $(DATA_FOLDER)/DataCPM \
				--pan_crops $(DATA_FOLDER)/pan_crops \
				--benchmark 0 \
				-resume


clean:
	rm environment/segmentation-unet.sif
	nextflow clean
	# maybe remove singularity image and clean up...?
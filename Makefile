
DATA_NUC=/data/dataset/nuc_segmentation
DATA_TISSUE=/data/dataset/tissue_segmentation/data_tissue

create_image:
	sudo bash environment/create-img.sh

tissue_local: src/tissue_segmentation_model.nf
	nextflow $< --data_path $(DATA_TISSUE) \
				--benchmark 0 \
				-resume

tissue: src/tissue_segmentation_model.nf
	nextflow $< --data_path $(DATA_TISSUE) \
				--benchmark 1 \
				-resume

benchmark: src/nuclei_segmentation_model.nf
	nextflow $< --tnbc_path $(DATA_NUC)/TNBC_NucleiSegmentation \
				--monuseg_path $(DATA_NUC)/MoNucleiSegmentation \
				--monusac_path $(DATA_NUC)/MoNuSAC_images_and_annotations \
				--consep_path $(DATA_NUC)/consep/CoNSeP \
				--cpm_path $(DATA_NUC)/DataCPM \
				--pancrops_path $(DATA_NUC)/pan_crops \
				--benchmark 1 \
				-resume

Unet_vgg: src/nuclei_segmentation_model.nf
	nextflow $< --tnbc_path $(DATA_NUC)/TNBC_NucleiSegmentation \
				--monuseg_path $(DATA_NUC)/MoNucleiSegmentation \
				--monusac_path $(DATA_NUC)/MoNuSAC_images_and_annotations \
				--consep_path $(DATA_NUC)/consep/CoNSeP \
				--cpm_path $(DATA_NUC)/DataCPM \
				--pancrops_path $(DATA_NUC)/pan_crops \
				--benchmark 0 \
				-resume

ENV=export PYTHONPATH=`pwd`/src/python/nn:`pwd`/src/templates/segmentation:$${PYTHONPATH}

transform_wsi_into_nuclei_table: src/from_wsi_to_nuclei_table.nf
	$(ENV); nextflow $< -resume

test_transform_wsi_into_nuclei_table: src/from_wsi_to_nuclei_table.nf
	$(ENV); nextflow $< --data "/data/dataset/camelyon2016/*/*/*.tif" -resume

clean:
	rm environment/segmentation-unet.sif
	nextflow clean
	# maybe remove singularity image and clean up...?